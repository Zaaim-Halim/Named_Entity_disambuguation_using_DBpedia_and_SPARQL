##### added by me to map urls ####
from django.urls import path
from . import views

urlpatterns = [
    path('',views.say_hello),
    path('disambiguate',views.serve_ajax)
]