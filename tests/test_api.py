"""API 层集成测试（TestClient，全离线）。"""


class TestUploadAndList:
    def test_upload_txt(self, api_client):
        resp = api_client.post(
            "/documents/upload",
            files={"file": ("制度.txt", "考勤制度：每天工作时间9点到18点。".encode("utf-8"), "text/plain")},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chunks"] >= 1
        assert data["total_chunks"] >= 1

    def test_upload_unsupported_type(self, api_client):
        resp = api_client.post(
            "/documents/upload",
            files={"file": ("a.docx", b"MZ...", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "不支持" in resp.json()["detail"]

    def test_upload_duplicate_rejected(self, api_client):
        payload = {"file": ("重复.txt", "相同内容".encode("utf-8"), "text/plain")}
        assert api_client.post("/documents/upload", files=payload).status_code == 200
        resp = api_client.post("/documents/upload", files=payload)
        assert resp.status_code == 409
        assert "已存在" in resp.json()["detail"]

    def test_list_documents(self, api_client):
        api_client.post(
            "/documents/upload",
            files={"file": ("制度.txt", "年假规定：入职满一年享5天带薪年假。".encode("utf-8"), "text/plain")},
        )
        resp = api_client.get("/documents")
        assert resp.status_code == 200
        assert resp.json()["documents"][0]["name"] == "制度.txt"

    def test_delete_document(self, api_client):
        api_client.post(
            "/documents/upload",
            files={"file": ("临时.txt", "临时内容".encode("utf-8"), "text/plain")},
        )
        resp = api_client.delete("/documents/临时.txt")
        assert resp.status_code == 200 and resp.json()["deleted"] is True
        assert api_client.get("/documents").json()["documents"] == []
        assert api_client.delete("/documents/不存在.txt").status_code == 404


class TestChat:
    def test_chat_without_docs_returns_400(self, api_client):
        resp = api_client.post("/chat", json={"question": "你好"})
        assert resp.status_code == 400
        assert "上传文档" in resp.json()["detail"]

    def test_chat_with_docs(self, api_client, monkeypatch):
        api_client.post(
            "/documents/upload",
            files={"file": ("制度.txt", "年假规定：入职满一年享5天带薪年假。".encode("utf-8"), "text/plain")},
        )
        from tests.conftest import FakeAgent

        monkeypatch.setattr(api_client.state_mod, "agent", FakeAgent())
        resp = api_client.post("/chat", json={"question": "年假几天", "conversation_id": "t1"})
        assert resp.status_code == 200
        assert "测试回答" in resp.json()["answer"]
        assert resp.json()["request_id"]

    def test_stream_chat(self, api_client, monkeypatch):
        from tests.conftest import FakeAgent

        api_client.post(
            "/documents/upload",
            files={"file": ("制度.txt", "年假规定：入职满一年享5天带薪年假。".encode("utf-8"), "text/plain")},
        )
        monkeypatch.setattr(api_client.state_mod, "agent", FakeAgent())
        with api_client.stream("POST", "/chat/stream", json={"question": "你好"}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert all(ch in body for ch in "测试流式")

    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"]
        assert data["config"]["api_key_configured"] is True

    def test_root_serves_frontend(self, api_client):
        resp = api_client.get("/")
        assert resp.status_code == 200
        assert "企业知识库" in resp.text or "助手" in resp.text