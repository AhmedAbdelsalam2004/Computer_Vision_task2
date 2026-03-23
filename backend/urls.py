from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView
from . import views

urlpatterns = [
    path("",                     RedirectView.as_view(url="http://localhost:5173/", permanent=False)),
    path("api/state/",           views.StateView.as_view(),          name="api-state"),
    path("api/upload/",          views.UploadView.as_view(),         name="api-upload"),
    path("api/undo/",            views.UndoView.as_view(),           name="api-undo"),
    path("api/reset/",           views.ResetView.as_view(),          name="api-reset"),
    path("api/detect-shapes/",   views.DetectShapesView.as_view(),   name="api-detect-shapes"),
    path("api/active-contour/",  views.ActiveContourView.as_view(),  name="api-active-contour"),
    path("api/switch-mode/",     views.SwitchModeView.as_view(),     name="api-switch-mode"),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)