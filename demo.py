from pathlib import Path
import cv2
import dlib
from os.path import basename
from pandas import DataFrame
import numpy as np
import argparse
from contextlib import contextmanager
from keras.utils.data_utils import get_file
from model import get_model


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="InceptionResNetV2",
                        help="model name: 'ResNet50' or 'InceptionResNetV2'")
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. age_only_weights.029-4.027-5.250.hdf5)")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield image_path, cv2.resize(img, (int(w * r), int(h * r)))


def main():
    args = get_args()
    model_name = args.model_name
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir

    if not weight_file:
        filename = 'weights.003-3.091-4.373.hdf5'
        
    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model = get_model(model_name=model_name)
    model.load_weights(filename)
    img_size = model.input.shape.as_list()[1]

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    data = []
    j=0
    for path , img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)
        j+=1
        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        print ("------d>",j,"\n")
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results.dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = str(int(predicted_ages[i]))
                draw_label(img, (d.left(), d.top()), label)

            rounded_age = int(np.round(predicted_ages))
            file_name = basename(str(path))
            original_age = int(file_name[4:6])
            diff =  original_age - rounded_age
            corr_age = rounded_age + ((((rounded_age-30) ** 2) / 100) - 9)
            corr_diff = original_age - corr_age
            if (diff<20):
                row = [file_name, original_age, rounded_age, diff, corr_age, corr_diff]
                data.append(row)
            print("\n------a>",j,"\n")    
                
    print("\n------c>",j,"\n")    

    print("\rCompleted")
    columns = ["file_name", "org_age", "est_age", "age_dif", "cor_age", "cor_dif"]
    df = DataFrame(data=data, columns=columns)
    df.to_csv("fgnet-test-4.csv")

    diff_col = np.abs(df.cor_dif)
    n = diff_col.shape[0]
    err = np.sum(diff_col) / n
    print("MAE:", err)
if __name__ == '__main__':
    main()