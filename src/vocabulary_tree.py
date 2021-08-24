import numpy as np
from sklearn.cluster import KMeans
import cv2
import joblib
import os
from src.utils import get_logger

logger = get_logger()


class VocabTreeNode(object):

    def __init__(self, val=None, children=None, num=0):
        """
        :param val: visual word 的向量
        :param children: 该结点的孩子结点们
        """
        self.val = val
        if children is None:
            self.children = []
        else:
            self.children = children

        # 该结点的编号
        self.num = num

        # trained KMeans model
        self.model = None

        # 储存每个经过该结点的image，并记录image的次数
        self.images = {}

        # 记录每个图片的description vector, 只有叶子结点会记录
        self.vectors = {}

        # 该结点在所有图片中出现的总次数
        self.total = 0

    def add_image(self, image_id):
        if image_id in self.images:
            self.images[image_id] += 1
        else:
            self.images[image_id] = 1

    def get_image_count(self, image_id):
        if image_id not in self.images:
            return 0
        return self.images[image_id]

    def get_total_count(self):
        """
        :return: 获取该结点在所有图片中出现的总次数
        """
        if self.total != 0:
            return self.total
        count = 0
        for image in self.images:
            count += self.images[image]
        self.total = count
        return self.total


class VocabTree(object):

    def __init__(self, k=None, l=None):
        """
        :param k: 树的宽度，也就是每个结点的孩子数的数量
        :param l: 树的高度
        """
        self.root = None
        self.curr_height = 0
        self.k = k
        self.max_height = l
        self.vocab_size = 0
        self.curr_num = -1
        self.image_count = 0

    def build_a_tree(self, image_key_points):
        X = None
        for image_id, kps in image_key_points:
            if X is None:
                X = kps
            else:
                X = np.concatenate([X, kps])
            self.image_count += 1

        logger.info("train data shape:{}".format(X.shape))
        root = self._build(X)
        setattr(self, "root", root)
        return root

    def _build(self, X):
        """
        BFS建树
        :param X:
        :param val:
        :return:
        """
        # print(X.shape, self.curr_height, self.max_height, self.k)
        root = VocabTreeNode(num=-1)
        queue = [(root, X)]

        self.curr_num = 0
        while queue:
            curr, data_x = queue[0]
            self.curr_height = self.get_height(root)
            if self.curr_height > self.max_height or data_x.shape[0] < self.k:
                break
            model = KMeans(n_clusters=self.k, init='k-means++')
            labels = model.fit_predict(data_x)
            curr.model = model
            visual_words = model.cluster_centers_
            # print(self.curr_height, self.max_height, data_x.shape, len(visual_words))

            for i, vw in enumerate(visual_words):
                child = VocabTreeNode(val=vw, num=self.curr_num)
                self.curr_num += 1
                indexes = labels == i
                curr.children.append(child)
                queue.append((child, data_x[indexes, :]))
            queue.pop(0)
        self.vocab_size = self.curr_num
        self.root = root
        return root

    def get_height(self, root=None):
        if root is None:
            root = self.root
        return self._height(root)

    def _height(self, root):
        if root is None:
            return 0

        max_height = 0
        for child in root.children:
            child_height = self._height(child)
            if child_height > max_height:
                max_height = child_height
        return max_height + 1

        # left_height = 0
        # right_height = 0
        # if root.left is not None:
        #     left_height = self._height(root.left)
        # if root.right is not None:
        #     right_height = self._height(root.right)

        # return max(left_height, right_height) + 1

    def serialize(self, root=None, K=None, model_dir='models/kmeans'):
        if not root:
            root = self.root
        if K is None:
            K = self.k
        queue = []
        ret = []
        queue.append(root)
        while queue:
            r = queue[0]
            queue.pop(0)
            if r == "#":
                ret.append("#")
                continue
            model_file = 'kmeans-{}.pkl'.format(r.num)
            model_file = os.path.join(model_dir, model_file)
            joblib.dump(r.model, model_file)
            ret.append([r.val, r.num, r.images, r.vectors, r.total, model_file])

            for i in range(K):
                if i >= len(r.children):
                    queue.append("#")
                else:
                    queue.append(r.children[i])

        while ret:
            if ret[-1] != "#":
                break
            del ret[-1]
        return ret
        # ret_str = ''
        # for s in ret:
        #     ret_str += str(s) + ","
        #
        # ret_str = "{" + ret_str.rstrip(',') + "}"
        #
        # return ret_str

    def deserialize(self, data):
        # write your code here
        if not data:
            return None

        # dataList = data.lstrip('{').rstrip('}').split(',')
        dataList = data
        queue = []
        root = VocabTreeNode(val=np.array(dataList[0][0]), num=dataList[0][1])
        root.images = dataList[0][2]
        root.vectors = {k: np.array(v) for k, v in dataList[0][3].items()}
        root.total = dataList[0][4]
        root.model = joblib.load(dataList[0][5])

        queue.append(root)

        child_i = 0
        index = 0
        for val, num, images, vectors, total, model_file in dataList[1:]:
            node = queue[index]
            new_node = VocabTreeNode(num=num, val=np.array(val))
            new_node.images = images
            new_node.vectors = {k: np.array(v) for k, v in vectors.items()}
            new_node.total = total
            new_node.model = joblib.load(model_file)

            node.children.append(new_node)
            child_i += 1
            if child_i == self.k:
                index += 1
                child_i = 0
            queue.append(new_node)
        self.vocab_size = index
        self.root = root
        return root

    def retrieval(self, key_points):
        """
        计算一个database之外的图片（检索图片）的represented vector
        :param X:
        :return: represented vector
        """
        mi_map = {}
        similar_images = self.pass_update_visual_words(key_points, mi_map=mi_map, return_similar=True)
        similar_images = set(similar_images)
        vector = self.transform(mi_map=mi_map)
        return vector, similar_images

    # 作用是所有的图片遍历一遍整棵树，记录visual word在每个图片的出现次数
    def pass_update_visual_words(self, key_points, image_id=None, mi_map=None, return_similar=False, weight_thre=0.1):
        similar_images = []
        for i, kp in enumerate(key_points):
            curr = self.root
            while curr:
                if mi_map is not None:
                    if curr.num in mi_map:
                        mi_map[curr.num] += 1
                    else:
                        mi_map[curr.num] = 0
                else:
                    curr.add_image(image_id)

                # 如果是叶子结点，则返回此叶子结点上所有权重大于阈值的images
                if not curr.children:
                    if return_similar:
                        for image_id in curr.images:
                            # total = sum(curr.images.values())
                            # if curr.images[image_id] / total > weight_thre:
                            similar_images.append(image_id)
                    break
                model = curr.model
                # 到所有中心的距离
                kp = kp.reshape(1, -1)
                dist = model.transform(kp)
                index = np.argmax(dist)
                curr = curr.children[index]
            # if i % 100 == 0:
            #     print("left key points:{}".format(len(key_points) - i + 1))
        return similar_images

    # 得到image的represented vector
    def transform(self, image_id=None, mi_map=None):
        assert image_id is not None or mi_map is not None

        vector = np.zeros(shape=(self.vocab_size,))
        queue = [self.root]
        while queue:
            curr = queue[0]
            queue.pop(0)
            if curr.num != -1:
                mi = 0
                if image_id and image_id in curr.images:
                    mi = curr.images[image_id]
                elif mi_map and curr.num in mi_map:
                    mi = mi_map[curr.num]
                vector[curr.num] = self.tfidf(mi, self.image_count, len(curr.images))

            for child in curr.children:
                queue.append(child)
        # 归一化为单位向量
        vector = vector/np.linalg.norm(vector)
        return vector

    def tfidf(self, mi, N, Ni):
        """
        :param mi: 该图片经过该visual word的所有description vectors的数量
        :param N: 数据库中image的总数
        :param Ni: 至少有一条description vector经过该点的图像数量
        :return:
        """
        wi = np.log(N/(Ni+1e-12))
        di = mi * wi
        return di


def my_test():
    img = cv2.imread('beautiful_girl.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT().create()
    # kp = sift.detect(gray, None)
    kp, des = sift.detectAndCompute(gray, None)
    print(des.shape)

    # 建立一棵树
    image_kps = [
        ("beautiful_girl", des),
    ]

    vt = VocabTree()
    vt.build_a_tree(image_kps, k=3, l=3)
    ret = vt.serialize(vt.root)
    print(ret)
    # # ret = vt.serialize(vt.deserialize(ret))
    # # print(ret)
    #
    # 图片遍历一遍整棵树，记录visual word在每个图片的出现次数
    vt.pass_update_visual_words(des, image_id='beautiful_girl')
    print(vt.serialize())

    # 统计database中每一个image的description vector
    vt.transform(image_id="beautiful_girl")
    print(vt.serialize())


def my_test_print():
    vt = VocabTreeNode(val=0, children=[])
    for i in range(3):
        vt.children.append(VocabTreeNode(i))
    for child in vt.children:
        for i in range(3):
            child.children.append(VocabTreeNode(i))

    tr = VocabTree()


if __name__ == '__main__':
    # my_test_print()
    my_test()


