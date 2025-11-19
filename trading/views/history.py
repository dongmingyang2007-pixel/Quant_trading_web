from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from ..history import delete_history_record


@require_POST
@login_required
def delete_history(request, record_id: str):
    success = delete_history_record(record_id, user_id=str(request.user.id))
    status_code = 200 if success else 404
    return JsonResponse({"success": success}, status=status_code)

