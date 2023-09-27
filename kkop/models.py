from django.db import models


class UploadPic(models.Model):
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to="kkop/uploads/")
    is_active = models.BooleanField(default=True)
    is_home = models.BooleanField(default=False)

class DetectionResult(models.Model):
    upload_pic = models.ForeignKey(UploadPic, on_delete=models.CASCADE)
    result_image = models.ImageField(upload_to='detection_results/')



