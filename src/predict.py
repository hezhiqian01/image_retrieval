import argparse
import joblib
import traceback
from src.train import get_database_images
import json
from src.utils import get_logger

logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="trained model file path")
    parser.add_argument("--images", type=str, required=True,
                        help="one image path or dir contains all images needed to retrieval")
    parser.add_argument("--output", type=str, default="prediction_result/result.txt", help="result output filename")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    try:
        model = joblib.load(args.model_path)
        logger.info("model loaded")

        result = []
        for image_id, kps in get_database_images(args.images):
            logger.info("started to predict {} ...".format(image_id))
            res = model.forward(kps)
            logger.info("predict {} finished".format(image_id))
            data = {}
            similar_images = []
            for img_id, dist in res:
                similar_images.append(img_id)
            data['image_id'] = image_id
            data['similar_images'] = similar_images[:10]
            result.append(data)

        logger.info("predict finished")

        with open(args.output, 'w') as f:
            f.write(json.dumps(result, indent=4))
        logger.info("")

    except Exception as e:
        logger.error(traceback.print_exc())


if __name__ == '__main__':
    main()
