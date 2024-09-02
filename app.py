from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from slippage_estimator import get_output_token, get_add_lp, get_remove_single_lp, get_remove_lp

app = Flask(__name__)
CORS(app)


@cross_origin(origins=["http://localhost:3000", "https://app.lucidly.finance"])
@app.route('/get_output_token', methods=['GET'])
async def get_output_token_route():
    _i = int(request.args.get('_i'))
    _j = int(request.args.get('_j'))
    _dx = int(request.args.get('_dx'))
    result = await get_output_token(_i, _j, _dx)
    return jsonify({'result': int(result)})


@cross_origin(origins=["http://localhost:3000", "https://app.lucidly.finance"])
@app.route('/get_add_lp', methods=['POST'])
async def get_add_lp_route():
    _amounts = request.json.get('_amounts')
    result = await get_add_lp(_amounts)
    return jsonify({'result': int(result)})


@cross_origin(origins=["http://localhost:3000", "https://app.lucidly.finance"])
@app.route('/get_remove_lp', methods=['GET'])
async def get_remove_lp_route():
    _lp_amount = int(request.args.get('_lp_amount'))
    result = await get_remove_lp(_lp_amount)
    return jsonify({'result': result})


@cross_origin(origins=["http://localhost:3000", "https://app.lucidly.finance"])
@app.route('/get_remove_single_lp', methods=['GET'])
async def get_remove_single_lp_route():
    _token = int(request.args.get('_token'))
    _lp_amount = int(request.args.get('_lp_amount'))
    result = await get_remove_single_lp(_token, _lp_amount)
    return jsonify({'result': int(result)})


if __name__ == '__main__':
    app.run(debug=True)
