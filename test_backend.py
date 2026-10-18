"""
Automated backend verification script for Campus Q&A Bot.
Tests upload, question answering, page citations, out-of-context threshold,
summary synthesis, and quiz generation.
"""
import io
import json
from app import app, CHUNKS

def run_tests():
    print("==========================================")
    print("Running Automated Tests for Campus Q&A Bot")
    print("==========================================")
    client = app.test_client()

    # Test 1: Get initial status
    print("\n[Test 1] Testing /status endpoint...")
    res = client.get("/status")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    status_data = res.get_json()
    print("Status response:", status_data)

    # Test 2: Upload PDF
    print("\n[Test 2] Testing /upload with DBMS_Unit3_Notes.pdf...")
    with open("DBMS_Unit3_Notes.pdf", "rb") as f:
        pdf_bytes = f.read()

    data = {
        "subject": "DBMS",
        "files": (io.BytesIO(pdf_bytes), "DBMS_Unit3_Notes.pdf")
    }
    res = client.post("/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 200, f"Upload failed: {res.get_data(as_text=True)}"
    upload_res = res.get_json()
    print("Upload result:", upload_res)
    assert upload_res["chunks_added"] > 0, "No chunks were added!"
    assert upload_res["total_chunks"] == len(CHUNKS)

    # Test 3: Ask a question about ACID properties
    print("\n[Test 3] Testing /ask with 'What are the ACID properties in DBMS?'...")
    res = client.post(
        "/ask",
        json={"question": "What are the ACID properties in DBMS?", "subject": "DBMS"}
    )
    assert res.status_code == 200, f"Ask failed: {res.get_data(as_text=True)}"
    ask_res = res.get_json()
    print("Ask response (Points):", json.dumps(ask_res.get("points"), indent=2))
    print("Ask response (Summary):", ask_res.get("summary"))
    print("Ask response (Sources):", ask_res.get("sources"))
    print("Ask response (Confidence):", ask_res.get("confidence"), "%")
    assert not ask_res.get("not_found"), "Expected question to be found in documents!"
    assert len(ask_res.get("sources", [])) > 0, "Expected at least one source citation!"
    assert any("p.1" in s for s in ask_res["sources"]), f"Expected page 1 citation for ACID, got {ask_res['sources']}"

    # Test 4: Ask a question about B+ Trees (Page 3)
    print("\n[Test 4] Testing /ask with 'Why are B+ trees efficient for range queries?'...")
    res = client.post(
        "/ask",
        json={"question": "Why are B+ trees efficient for range queries?", "subject": "DBMS"}
    )
    assert res.status_code == 200
    bplus_res = res.get_json()
    print("B+ Tree Sources:", bplus_res.get("sources"))
    print("B+ Tree Confidence:", bplus_res.get("confidence"), "%")
    assert any("p.3" in s for s in bplus_res["sources"]), f"Expected page 3 citation for B+ Tree, got {bplus_res['sources']}"

    # Test 5: Out of context question (Threshold < 0.30)
    print("\n[Test 5] Testing out-of-context threshold with 'How do you bake sourdough bread?'...")
    res = client.post(
        "/ask",
        json={"question": "How do you bake sourdough bread with yeast?", "subject": "DBMS"}
    )
    assert res.status_code == 200
    out_res = res.get_json()
    print("Out-of-context response:", out_res)
    assert out_res.get("not_found") is True, f"Expected not_found=True for unrelated question! Got {out_res}"
    assert "Not found" in out_res.get("summary", ""), "Expected 'Not found' in summary!"

    # Test 6: Summary endpoint
    print("\n[Test 6] Testing /summary endpoint...")
    res = client.post("/summary", json={"subject": "DBMS"})
    assert res.status_code == 200, f"Summary failed: {res.get_data(as_text=True)}"
    summary_res = res.get_json()
    print("Summary Key Points Count:", len(summary_res.get("key_points", [])))
    print("Summary Overview:", summary_res.get("summary"))
    assert len(summary_res.get("key_points", [])) >= 3

    # Test 7: Quiz endpoint
    print("\n[Test 7] Testing /quiz endpoint...")
    res = client.post("/quiz", json={"subject": "DBMS"})
    assert res.status_code == 200, f"Quiz failed: {res.get_data(as_text=True)}"
    quiz_res = res.get_json()
    questions = quiz_res.get("questions", [])
    print(f"Generated {len(questions)} quiz questions.")
    assert len(questions) > 0, "No questions returned!"
    first_q = questions[0]
    print("Sample question:", json.dumps(first_q, indent=2))
    assert "question" in first_q
    assert "options" in first_q and len(first_q["options"]) == 4
    assert "correct_answer" in first_q
    assert "explanation" in first_q
    assert "source" in first_q

    print("\n==========================================")
    print("ALL BACKEND AUTOMATED TESTS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    run_tests()
