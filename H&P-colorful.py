# cv2 = library OpenCV yang digunakan untuk membaca gambar, mengolah gambar, dan 
# menjalankan model colorization.
# numpy = library untuk mengolah array (angka), dipakai untuk memproses data warna.

import cv2 as cv
import numpy as np



#.prototxt → arsitektur jaringan neural.
# .caffemodel → bobot model (hasil training).
# .npy → titik warna (cluster warna) yang dipakai model.
proto = "colorization_deploy_v2.prototxt"
model = "colorization_release_v2.caffemodel"
pts = "pts_in_hull.npy"

# readNetFromCaffe → memuat model AI Caffe ke OpenCV.
# np.load() → memuat data cluster warna dari file .npy.

net = cv.dnn.readNetFromCaffe(proto, model)
pts_in_hull = np.load(pts)

# Mengubah bentuk matriks pts_in_hull menjadi format yang model butuhkan:
# 2 channel warna (a dan b)
# 313 titik warna (cluster)
# bentuk 4D untuk layer Caffe
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

# "Ini loh kumpulan warna yang bisa kamu gunakan."
# Layer yang bernama class8_ab adalah layer yang memprediksi warna
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]

# mengatur intensitas warna
# angka 2.606 adalah nilai standard dari model aslinya.
# Tanpa ini, warna output jadi pudar.
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]


# cv.imread() → membaca file gambar hitam putih / sepia.
# img.shape[:2] → mengambil tinggi (h) dan lebar (w) gambar untuk nanti.
img = cv.imread("img-3.jpg")
h, w = img.shape[:2]


# Mengubah nilai piksel dari 0–255 menjadi 0–1 agar sesuai input neural network.
img_rgb = (img.astype(np.float32) / 255.0)

# Mengubah gambar dari:
# BGR → LAB
# Kenapa LAB?
# LAB memisahkan L = luminance (terang/gelap)
# dan a,b = warna
# Model hanya butuh L sebagai input.
img_lab = cv.cvtColor(img_rgb, cv.COLOR_BGR2LAB)

# Mengambil channel L (kecerahan) saja untuk diproses oleh model AI.
L = img_lab[:, :, 0]

# Resize to network input size
input_L = cv.resize(L, (224, 224))
input_L -= 50  # normalize

net.setInput(cv.dnn.blobFromImage(input_L))
ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize color result back to original size
ab_dec_us = cv.resize(ab_dec, (w, h))

# Combine L + ab channels
lab_out = np.concatenate((L[:, :, np.newaxis], ab_dec_us), axis=2)
img_out = cv.cvtColor(lab_out, cv.COLOR_LAB2BGR)
img_out = np.clip(img_out, 0, 1)

# Save result
cv.imwrite("rgb_mod.png", (img_out * 255).astype(np.uint8))

print("Selesai! → foto_modern_2025.jpg")
