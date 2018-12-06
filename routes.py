from flask import Flask, jsonify, request
import ocr


app = Flask(__name__)
ocr_data = ocr.ocr


@app.route("/")
def root():
    return jsonify({"message": "올바르지 않은 접근입니다.", "code": 0}), 400


@app.route("/ocr/ranking", methods=["POST"])
def get():
    print(request.form)
    if request.form.get("url"):
        img = ocr.get_image_from_url(request.form.get("url"))
    else:
        return jsonify({"message": "모든 인자를 입력해주세요.", "code": 101})
    img = ocr.img_normalize(img)
    per = ocr_data.check(ocr.get_per_box(img))
    score = ocr_data.check(ocr.get_score_box(img))
    if not 0 < per < 101:
        return jsonify({"message": "퍼센트 인식에 오류가 발생하였습니다.", "code": 201})

    if per and score:
        return jsonify({"per": per, "score": score})
    elif not per and score:
        return jsonify({"message": "이미지에 퍼센트 값이 없습니다.", "code": 202, "score": score})
    else:
        return jsonify({"message": "이미지에서 값을 구할 수 없습니다.", "code": 203})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
