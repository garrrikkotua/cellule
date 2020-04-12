from flask import Flask
from flask import request, render_template, redirect, url_for, flash
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from model import FCRN_A, UNet
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch
import math
import utils

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
TYPE1 = 'FCRN'
TYPE2 = 'UNet'

app = Flask(__name__)
app.secret_key = b'here should be some secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


global net
net = FCRN_A(N=2)
dic = torch.load('cell_FCRN_A.pth', map_location=torch.device('cpu'))
dic = {k[k.find('.') + 1:]: v for k, v in dic.items()}
net.load_state_dict(dic)

global u_net
u_net = UNet()
dic = torch.load('cell_UNet.pth', map_location=torch.device('cpu'))
dic = {k[k.find('.') + 1:]: v for k, v in dic.items()}
u_net.load_state_dict(dic)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(404)
def error_404(e):
    return render_template('page_404.html'), 404


@app.errorhandler(500)
def error_500(e):
    return render_template('page_505'), 500


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Вы не выбрали файл')
            return render_template('index.html', behaviour='auto')
        file = request.files['file']
        if file.filename == '':
            flash('Вы не выбрали файл')
            return render_template('index.html', behaviour='auto')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if utils.validate_image_size(filename):
                return redirect(url_for(request.form["inlineRadioOptions"], filename=file.filename))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Размер изображения слишком большой! Максимальный размер 700x700.')
            return render_template('index.html', behaviour='auto')
        else:
            flash('Недопустимый формат файла. Загружать можно только такие файлы: ' + ', '.join(ALLOWED_EXTENSIONS))
            return render_template('index.html', behaviour='auto')
    return render_template('index.html', behaviour='smooth')


@app.route('/thanks', methods=['GET'])
def thahks():
    return render_template('thanks.html')


@app.route('/fcrn/<filename>', methods=['GET', 'POST'])
def fcrn(filename):
    if request.method == "POST":
        # сохраняем рейтинг
        utils.save_rating(request.form['whatever1'], TYPE1)
        return redirect(url_for('thahks'))
    # Step 1 - image preparation
    my_image = Image.open(app.config['UPLOAD_FOLDER'] + filename)
    my_image = my_image.convert('RGB')
    my_image = transforms.ToTensor()(my_image)
    my_image = my_image.unsqueeze(0) / 255

    # Step 2 - generating density map
    d_map = net(my_image)
    path_to_save = os.path.join('static/maps', Path(filename).stem + '_predicted_fcrn.png')
    save_image(d_map, path_to_save)
    cell_count = float(d_map.sum()) / 100

    # Step 3 - returning prediction
    prediction = dict()
    prediction['cells'] = int(cell_count)
    prediction['predicted'] = Path(filename).stem + '_predicted_fcrn.png'
    prediction['original'] = filename
    prediction['type'] = TYPE1

    # Step 4
    return render_template('predict.html', prediction=prediction)


@app.route('/unet/<filename>', methods=['GET', 'POST'])
def unet(filename):
    # сохрагяем рейтинг
    if request.method == "POST":
        utils.save_rating(request.form['whatever1'], TYPE2)
        return redirect(url_for('thahks'))

    my_image = Image.open(app.config['UPLOAD_FOLDER'] + filename)
    my_image = my_image.convert('RGB')
    width, height = my_image.size
    width = 2 ** int(math.log(width, 2))
    height = 2 ** int(math.log(height, 2))
    my_image = transforms.Resize((width, height))(my_image)
    my_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    my_image = transforms.ToTensor()(my_image)
    my_image = my_image.unsqueeze(0) / 255

    d_map = u_net(my_image)
    path_to_save = os.path.join('static/maps', Path(filename).stem + '_predicted_unet.png')
    save_image(d_map, path_to_save)
    cell_count = float(d_map.sum()) / 100

    # Step 3 - returning prediction
    prediction = dict()
    prediction['cells'] = int(cell_count)
    prediction['predicted'] = Path(filename).stem + '_predicted_unet.png'
    prediction['original'] = filename
    prediction['type'] = TYPE2

    # Step 4
    return render_template('predict.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
