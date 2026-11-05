import socket

s = socket.create_connection(("127.0.0.1", 8155), timeout=25)
s.sendall(b"GET /api/runs/smoke-emiru/events HTTP/1.1\r\nHost: x\r\nAccept: text/event-stream\r\nConnection: close\r\n\r\n")
s.settimeout(12)
data = b""
closed = False
try:
    while True:
        chunk = s.recv(65536)
        if not chunk:
            closed = True
            break
        data += chunk
except socket.timeout:
    pass
print("bytes:", len(data), flush=True)
print("closed-by-server:", closed, flush=True)
head, _, _ = data.partition(b"\r\n\r\n")
print("HEAD:", head.decode("utf8", "replace")[:400], flush=True)
print("TAIL:", data[-160:].decode("utf8", "replace"), flush=True)

