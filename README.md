<h1>Cellule.ru</h1>

<p>This is repository for website Cellule.ru. On the website you can estimate number of cells on image (biological cells). It's a routine task for many biologists around the globe. However, sometimes it's very hard to count them all with a naked eye. So, this is the tool to help them. Of course, I don't guarantee it's 100% correct. At least, it's a good estimation.</p>

<p>The core of the app is convolutional neural net written in Pytorch. It was trained on synthetic data and tested on real one. Neural net shows very reliable results on regularly shaped even sized cells. The code for neural net and syhthetic data generating is taken from this repo: https://github.com/NeuroSYS-pl/objects_counting_dmap.</p>

<p>If you want to use pretrained models yourself, you can take it from my repo: <strong>cell_FCRN_A.pth and cell_UNet.pth</strong> - Pytorch saved models.</p>
