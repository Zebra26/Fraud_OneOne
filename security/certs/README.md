Development TLS certificates

This folder is intended for self-signed certificates used only in development.

Quickstart (OpenSSL):
- Generate CA: openssl req -x509 -new -nodes -keyout ca.key -sha256 -days 365 -out ca.crt -subj "/CN=DevCA"
- Generate server key: openssl genrsa -out server.key 2048
- Generate CSR: openssl req -new -key server.key -out server.csr -subj "/CN=localhost"
- Sign server cert: openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -sha256

Files used by the stack:
- server.crt, server.key: presented by services when ENABLE_TLS=true
- ca.crt: used by clients to validate TLS

Never commit real certificates or private keys. Use Vault/K8s/Docker secrets in production.

