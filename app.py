from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from slippage_estimator import get_output_token, get_add_lp, get_remove_single_lp, get_remove_lp
import asyncio

app = Flask(__name__)
CORS(app)


@cross_origin(origins=["http://localhost:3000", "https://app.lucidly.finance"])
@app.route('/get_output_token', methods=['GET'])
def get_output_token_route():
    _i = int(request.args.get('_i'))
    _j = int(request.args.get('_j'))
    _dx = int(request.args.get('_dx'))
    try:
        result = asyncio.run(get_output_token(_i, _j, _dx))
        return jsonify({'result': int(result)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400


@cross_origin(origins=["http://localhost:3000", "https://app.lucidly.finance"])
@app.route('/get_add_lp', methods=['POST'])
def get_add_lp_route():
    _amounts = request.json.get('_amounts')

    try:
        result = asyncio.run(get_add_lp(_amounts))
        return jsonify({'result': int(result)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400


@cross_origin(origins=["http://localhost:3000", "https://app.lucidly.finance"])
@app.route('/get_remove_lp', methods=['GET'])
def get_remove_lp_route():
    _mvt_amount = int(request.args.get('_mvt_amount'))
    try:
        result = asyncio.run(get_remove_lp(_mvt_amount))
        return jsonify({'result': result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400


@cross_origin(origins=["http://localhost:3000", "https://app.lucidly.finance"])
@app.route('/get_remove_single_lp', methods=['GET'])
def get_remove_single_lp_route():
    _token = int(request.args.get('_token'))
    _mvt_amount = int(request.args.get('_mvt_amount'))
    try:
        result = asyncio.run(get_remove_single_lp(_token, _mvt_amount))
        return jsonify({'result': int(result)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400


#  if __name__ == '__main__':
#      app.run(debug=True)
