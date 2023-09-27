from ultralytics import YOLO
import os
from django.core.files import File
import django
from kokten import settings
from kkop.models import UploadPic, DetectionResult

def run_detection():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kokten.settings")
    django.setup()

    latest_upload = UploadPic.objects.order_by('-id').first()

    if latest_upload is not None:
        latest_image_path = os.path.join(settings.MEDIA_ROOT, str(latest_upload.image))

        infer = YOLO("detection.pt FOLDER IN HERE")
        result = infer.predict(latest_image_path, save=True)

        detect_dir = os.path.join(settings.BASE_DIR, 'runs', 'detect')
        predict_dirs = [d for d in os.listdir(detect_dir) if
                        os.path.isdir(os.path.join(detect_dir, d)) and d.startswith('predict')]


        latest_predict_dir = max(predict_dirs, key=lambda x: int(x.lstrip('predict')) if x.lstrip(
            'predict').isdigit() else 0) if predict_dirs else None


        latest_upload = UploadPic.objects.order_by('-id').first()
        if latest_upload is not None:
            detection_result = DetectionResult(upload_pic=latest_upload)


            prediction_files = os.listdir(os.path.join(detect_dir, latest_predict_dir))


            for file_name in prediction_files:
                with open(os.path.join(detect_dir, latest_predict_dir, file_name), 'rb') as image_file:
                    detection_result.result_image.save(file_name, File(image_file))
                    detection_result.result_image.save(file_name, File(image_file))

            detection_result.save()
        else:
            print("Check your upload file format.")

