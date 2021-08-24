import cv2
import os
from src.vocabulary_tree import VocabTree
import json
import types
import datetime
import itertools
import numpy as np
from json import JSONEncoder
import joblib
import argparse
from src.utils import get_logger

logger = get_logger()


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_database_images(image_dir="train_images"):
    file_list = []
    if os.path.isfile(image_dir):
        file_list.append(image_dir)
    else:
        for filename in os.listdir(image_dir):
            img_filename = os.path.join(image_dir, filename)
            file_list.append(img_filename)

    for img_filename in file_list:
        img_id = os.path.basename(img_filename)
        img = cv2.imread(img_filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT().create()
        # kp = sift.detect(gray, None)
        kp, des = sift.detectAndCompute(gray, None)
        yield img_id, des


class ImageRetrievalModel(object):

    def __init__(self, k=3, l=2):
        self.vocab_tree = VocabTree(k, l)
        self.represented_vector = {}

    def train(self, image_key_points):
        """
        :param image_key_points: 可迭代对象, 第一个元素是image_id, 第二个是key_points
        :return:
        """

        assert isinstance(image_key_points, types.GeneratorType)

        logger.info("train started at {}".format(datetime.datetime.now()))

        image_key_points, bk1 = itertools.tee(image_key_points)
        bk1, bk2 = itertools.tee(bk1)
        # 建立一颗树
        self.vocab_tree.build_a_tree(image_key_points)
        logger.info("vocab tree built.")

        # 图片遍历一遍整棵树，记录visual word在每个图片的出现次数
        for image_id, kps in bk1:
            self.vocab_tree.pass_update_visual_words(kps, image_id)

        logger.info("visual words updated.")

        # 统计database中每一个image的represented vector

        for image_id, _ in bk2:
            self.represented_vector[image_id] = self.vocab_tree.transform(image_id)
        logger.info("represented vectors generated.")

        logger.info("train finished at {}".format(datetime.datetime.now()))

    def save_model(self, model_path="models"):
        save_data = {}
        kmeans_model_dir = os.path.join(model_path, "kmeans")
        if not os.path.isdir(kmeans_model_dir):
            os.mkdir(kmeans_model_dir)
        serialize_data = self.vocab_tree.serialize(model_dir=kmeans_model_dir)
        save_data['vocab_tree'] = serialize_data
        save_data['represented_vector'] = self.represented_vector
        save_data['vocab_size'] = self.vocab_tree.vocab_size
        save_data['image_count'] = self.vocab_tree.image_count
        save_data = json.dumps(save_data, cls=NumpyArrayEncoder)

        model_file = os.path.join(model_path, 'ImageRetrieval.json')
        with open(model_file, 'w') as f:
            f.write(save_data)
        logger.info("model saved in {}".format(model_file))

    def load_model(self, model_path="models/ImageRetrieval.json"):
        assert model_path.endswith("json") and os.path.isfile(model_path)
        with open(model_path, 'r') as f:
            data = json.loads(f.read())
        self.vocab_tree.deserialize(data['vocab_tree'])
        self.represented_vector = data['represented_vector']
        self.vocab_tree.vocab_size = data['vocab_size']
        self.vocab_tree.image_count = data['image_count']
        logger.info("model:{} loaded.".format(model_path))

    def forward(self, X):
        assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == 128
        r_vec, similar_images = self.vocab_tree.retrieval(X)
        # print(r_vec)
        # print(similar_images)
        # 获取该向量与数据库中的向量的距离
        dist = self.get_distance(r_vec, similar_images)
        dist = sorted(dist.items(), key=lambda x: x[1], reverse=False)
        return dist

    def get_distance(self, q, similar_images):
        dist = {}
        for image_id in similar_images:
            d = self.represented_vector[image_id]
            dist[image_id] = np.linalg.norm(q) + np.linalg.norm(d) + np.linalg.norm(q - d)
        return dist


def train(image_dir=None, k=10, l=5, model_output="models"):
    """
    train a vocab tree
    :return:
    """
    model = ImageRetrievalModel(k=k, l=l)
    train_data = get_database_images(image_dir)
    model.train(train_data)
    # model.save_model(model_output)
    joblib.dump(model, os.path.join(model_output, "model.pkl"))
    # print(model.vocab_tree.serialize())

    # model = ImageRetrievalModel(k=3, l=2)
    # model.load_model()
    # # print(model.vocab_tree.serialize())
    #
    # image_id = "word_1130.png"
    # # print(model.represented_vector)
    # for img_id, kps in get_database_images():
    #     if image_id == img_id:
    #         dist = model.forward(kps)
    #         print(dist)
    #         break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="train_images", help="train images dir")
    parser.add_argument("-k", type=int, default=10, help="how many children of one vocab tree node")
    parser.add_argument("-l", type=int, default=5, help="the depth of vocab tree")
    parser.add_argument("--model_output", type=str, default="models", help="model output dir")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train(image_dir=args.image_dir, k=args.k, l=args.l, model_output=args.model_output)


if __name__ == '__main__':
    main()
