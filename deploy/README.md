# 部署指南（生产环境）

以下提供三种可选方案：

1) Docker + 反向代理（Nginx/Traefik）
- 构建镜像：
  - `cd quant_trading_web`
  - `docker build -t quant-web .`
- 运行容器（使用 .env）：
  - `docker run --env-file .env -p 8000:8000 quant-web`
- 反向代理：将 `https://yourdomain.com` 转发到 `http://127.0.0.1:8000`，并开启 HTTPS。
- 静态资源由 WhiteNoise 提供，无需单独 Nginx 静态路径。

2) 服务器自建（Systemd + Gunicorn + Nginx）
- 服务器上：
  - `python -m venv venv && source venv/bin/activate`
  - `pip install -r requirements.txt`
  - `python manage.py migrate && python manage.py collectstatic --noinput`
  - 测试：`gunicorn -c gunicorn.conf.py quant_trading_site.wsgi:application`
- Nginx 反代示例（/deploy/nginx.sample.conf）：
  - `server_name yourdomain.com;` 反向代理到 `127.0.0.1:8000`。
  - 开启 TLS（letsencrypt certbot）。
- Systemd 服务示例（/deploy/gunicorn.service）：将 Gunicorn 常驻运行。

3) Render / Railway / Fly.io（云平台）
- Start command：`gunicorn -c gunicorn.conf.py quant_trading_site.wsgi:application`
- Build command：`pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
- 环境变量：
  - `DJANGO_DEBUG=0`
  - `DJANGO_ALLOWED_HOSTS=yourdomain.com`
  - `DJANGO_SECRET_KEY=...`（高强度）
  - 邮件相关 `EMAIL_*`，以及 `DJANGO_DEFAULT_FROM_EMAIL`
  - 如绑定自定义域名，设置 `DJANGO_CSRF_TRUSTED_ORIGINS=https://yourdomain.com`

重要环境变量
- `DJANGO_DEBUG=0`：生产必须关闭 DEBUG
- `DJANGO_ALLOWED_HOSTS`：逗号分隔域名/IP
- `DJANGO_SECRET_KEY`：随机字符串
- `DJANGO_CSRF_TRUSTED_ORIGINS`：例如 `https://yourdomain.com`
- 邮件：`EMAIL_HOST/PORT/USER/PASSWORD`、`EMAIL_USE_TLS` 或 `EMAIL_USE_SSL`

数据库
- 默认 SQLite（`db.sqlite3`）。初期访问量小可用。
- 若需 Postgres，可用云平台托管；把 `DATABASES['default']` 换成环境变量驱动（例如 dj-database-url）。

健康检查 / 监控
- Gunicorn 访问/错误日志输出到 stdout，建议通过进程管理或平台收集。
- 反代开启 gzip、HTTP/2。

安全小贴士
- 始终使用 HTTPS；若位于代理后，已在 settings 中设置 `SECURE_PROXY_SSL_HEADER`。
- 生产中设置：`SESSION_COOKIE_SECURE=1`、`CSRF_COOKIE_SECURE=1`（在 settings 已支持 ENV 控制）。
- 只把 `.env.example` 提交到仓库，`.env` 请勿提交。

