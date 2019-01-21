# Jerry Jia [11/30/2018] Enabled threshold of probobility, change to use 20180402-114759-CASIA-WebFace//20180402-114759.pb SavedModel and set GPU gpu_memory_fraction = 0.4
# Jerry Jia [01/21/2019] Changed input:0 to batch_join:0 for embedding and set GPU gpu_memory_fraction = 0.3

# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face
import facenet
import time

facenet_model_checkpoint = os.path.dirname(__file__) + "//..//model//20180402-114759-CASIA-WebFace//20180402-114759.pb"
#facenet_model_checkpoint = os.path.dirname(__file__) + "//..//model//20180402-114759-CASIA-WebFace" #checkpoint
classifier_model = os.path.dirname(__file__) + "//my_classifier_180731.pkl"

debug = False
name_threshold = 0.00001 # threshold of probability return
gpu_memory_fraction = 0.3

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        if debug:
            print("start find_faces in face.py")
        faces = self.detect.find_faces(image)
        #jjia
        start_time = time.time()
        if debug:
            print("finish find_faces in face.py")
        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            if debug:
                print("finish generate_embedding in face.py")
            face.name = self.identifier.identify(face)
            if debug:
                print("finish identify in face.py")
        #if debug:print("indentify,"+ str(time.time() - start_time))
        return faces


class Identifier:
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            if best_class_probabilities > name_threshold:
                #if debug:print (self.class_names[best_class_indices[0]],best_class_probabilities)
                return self.class_names[best_class_indices[0]]
            else:
                return None
            #return self.class_names[best_class_indices[0]]


class Encoder:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        #gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = self.sess.graph.get_tensor_by_name("batch_join:0") #jjia changed 2018/01/21
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        #jjia
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 50  # minimum size of face JJIA, original 20
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            #jjia
            #gpu_options = tf.GPUOptions(allow_growth=True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []
        start_time = time.time()
        #if debug:
        #    print("start detect_face in find_faces func")
        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)

        #if debug:
        #    print("finish detect_face in find_faces func")
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)
        #if debug:print("Detect_face," + str(time.time() - start_time))
        return faces
