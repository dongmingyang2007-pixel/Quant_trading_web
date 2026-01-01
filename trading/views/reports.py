from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.utils.translation import gettext as _

from ..history import get_history_record
from ..reporting import (
    ReportRenderingError,
    build_report_context,
    load_snapshot,
    render_report_csv,
    render_report_html,
    render_report_pdf,
)

SUPPORTED_FORMATS = {"json", "html", "pdf", "csv"}


@login_required
def export_report(request):
    history_id = request.GET.get("history_id")
    snapshot: dict[str, Any] = {}
    if history_id:
        record = get_history_record(history_id, user_id=str(request.user.id))
        if not record or not record.get("snapshot"):
            return HttpResponse(
                _("未找到对应的历史回测记录，请重新选择后再导出。"),
                status=404,
                content_type="text/plain; charset=utf-8",
            )
        snapshot = load_snapshot(record.get("snapshot"))
    else:
        raw_snapshot: Any = request.session.get("last_result")
        if not raw_snapshot:
            return HttpResponse(
                _("暂无可导出的回测报告，请先生成一次策略分析。"),
                status=404,
                content_type="text/plain; charset=utf-8",
            )
        snapshot = load_snapshot(raw_snapshot)
    if not snapshot:
        return HttpResponse(
            _("当前回测数据无法解析，请重新生成一次策略分析。"),
            status=400,
            content_type="text/plain; charset=utf-8",
        )

    fmt = (request.GET.get("format") or "json").strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        return HttpResponse(
            _("不支持的导出格式。可选：json/html/pdf/csv。"),
            status=400,
            content_type="text/plain; charset=utf-8",
        )

    context = build_report_context(snapshot, user=request.user)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename_base = f"quant_report_{timestamp}"

    if fmt == "json":
        payload = json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)
        response = HttpResponse(payload, content_type="application/json; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="{filename_base}.json"'
        return response

    if fmt == "html":
        html = render_report_html(context)
        response = HttpResponse(html, content_type="text/html; charset=utf-8")
        response["Content-Disposition"] = f'inline; filename="{filename_base}.html"'
        return response

    if fmt == "csv":
        csv_payload = render_report_csv(context)
        response = HttpResponse(csv_payload, content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="{filename_base}.csv"'
        return response

    try:
        pdf = render_report_pdf(context)
        response = HttpResponse(pdf, content_type="application/pdf")
        response["Content-Disposition"] = f'attachment; filename="{filename_base}.pdf"'
        return response
    except ReportRenderingError:
        html = render_report_html(context)
        response = HttpResponse(html, content_type="text/html; charset=utf-8")
        response["Content-Disposition"] = f'inline; filename="{filename_base}.html"'
        response["X-Export-Fallback"] = "html"
        return response
