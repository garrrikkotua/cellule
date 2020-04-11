from datetime import datetime
import os
from PIL import Image

PATH = os.path.join('data', 'ratings.csv')

MAX_SIZE = 700
UPLOAD_FOLDER = 'static/uploads/'


def validate_image_size(filename):
    my_image = Image.open(os.path.join(UPLOAD_FOLDER, filename))
    width, height = my_image.size
    if width > MAX_SIZE or height > MAX_SIZE:
        return False
    return True


def init_csv():
    if os.path.exists(PATH):
        return
    with open(PATH, 'w') as f:
        f.write('Timestamp, ' + 'Net, ' + 'Rating')
        f.write('\n')


def save_rating(value, net: str):
    init_csv()
    with open(PATH, 'a') as f:
        f.write('{}, {}, {}'.format(datetime.now(), net, value))
        f.write('\n')

