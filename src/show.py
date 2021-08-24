import matplotlib.pyplot as plt
from matplotlib import cm
import json
import cv2

a = set()
str.lstrip()

def main():

    with open("prediction_result/result.txt", "r") as f:
        data = json.loads(f.read())
        for res in data:
            ori_image = res["image_id"]
            print(ori_image)
            plt.figure()
            plt.title("image retrieval result")
            plt.subplot(3, 3, 1)
            plt.imshow(cv2.cvtColor(cv2.imread("train_images/" + ori_image), cv2.COLOR_BGR2RGB))
            plt.xticks([])
            plt.yticks([])
            plt.title("retrieved image")
            for i, sim in enumerate(res['similar_images']):
                if i > 2:
                    break
                plt.subplot(3, 3, i+2)
                plt.imshow(cv2.cvtColor(cv2.imread("train_images/" + sim), cv2.COLOR_BGR2RGB))
                plt.title("similar image {}".format(i+1))
                plt.xticks([])
                plt.yticks([])
            plt.show()


if __name__ == '__main__':
    main()
