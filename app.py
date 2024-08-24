from flask import Flask, request, jsonify
from slippage_estimator import get_output_token, get_add_lp, get_remove_single_lp

app = Flask(__name__)


@app.route('/get_output_token', methods=['GET'])
def get_output_token_route():
    _i = int(request.args.get('_i'))
    _j = int(request.args.get('_j'))
    _dx = int(request.args.get('_dx'))
    result = int(get_output_token(_i, _j, _dx))
    return jsonify({'result': result})


@app.route('/get_add_lp', methods=['POST'])
def get_add_lp_route():
    _amounts = request.json.get('_amounts')
    result = int(get_add_lp(_amounts))
    return jsonify({'result': result})


@app.route('/get_remove_single_lp', methods=['GET'])
def get_remove_single_lp_route():
    _token = int(request.args.get('_token'))
    _lp_amount = int(request.args.get('_lp_amount'))
    result = int(get_remove_single_lp(_token, _lp_amount))
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
