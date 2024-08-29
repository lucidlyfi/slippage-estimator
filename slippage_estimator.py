from web3 import AsyncWeb3
import asyncio
from typing import List
import json

try:
    profile
except NameError:

    def profile(func):
        return func


# loading web3 instance
rpc_url = "https://eth.merkle.io"
# rpc_url = "http://127.0.0.1:8545"
# rpc_url = "https://eth.rpc.blxrbdn.com"
web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))

# loading pool abi
f = open("abi_data/Pool.sol.json")
data = json.load(f)
pool_abi = data["abi"]

#  loading mastervault abi
f1 = open("abi_data/MasterVault.sol.json")
data1 = json.load(f1)
mastervault_abi = data1["abi"]

# pool contract address
pool_address = "0x8dBE744F6558F36d34574a0a6eCA5A8dAa827235"
pool = web3.eth.contract(address=pool_address, abi=pool_abi)

# rate provider abi
rate_provider_abi = '[{"inputs":[],"name":"RateProvider__InvalidParams","type":"error"},{"inputs":[{"internalType":"address","name":"token_","type":"address"}],"name":"rate","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]'

tokens = [
    "0xD9A442856C234a39a81a089C06451EBAa4306a72",
    "0xEEda34A377dD0ca676b9511EE1324974fA8d980D",
    "0x39F5b252dE249790fAEd0C2F05aBead56D2088e1",
    "0xeC3B2CC4C6a8fC9a13620A91622483b56E2E6fD9",
]

rate_providers = [
    "0xC4EF2c4B4eD79CD7639AF070d4a6A82eEF5edd4f",
    "0xC4EF2c4B4eD79CD7639AF070d4a6A82eEF5edd4f",
    "0x60d4BCab4A8b1849Ca19F6B4a6EaB26A66496267",
    "0x3C730BC8Ff9d7D51395c180c409597ae80A63056",
]

#  math stuff
PRECISION = 1_000_000_000_000_000_000
MAX_NUM_ASSETS = 32

WEIGHT_SCALE = 1_000_000_000_000
WEIGHT_MASK = 2**20 - 1
TARGET_WEIGHT_SHIFT = -20
LOWER_BAND_SHIFT = -40
UPPER_BAND_SHIFT = -60

# Powers of 10
E3 = 1_000
E6 = E3 * E3
E9 = E3 * E6
E12 = E3 * E9
E15 = E3 * E12
E17 = 100 * E15
E18 = E3 * E15
E20 = 100 * E18
E36 = E18 * E18

MAX_POW_REL_ERR = 100  # 1e-16
MIN_NAT_EXP = -41 * E18
MAX_NAT_EXP = 130 * E18
LOG36_LOWER = E18 - E17
LOG36_UPPER = E18 + E17
MILD_EXP_BOUND = 2**254 // 100_000_000_000_000_000_000

# x_n = 2^(7-n), a_n = exp(x_n)
# Values in 20 decimals for n >= 2
X0 = 128 * E18  # 18 decimals
A0 = 38_877_084_059_945_950_922_200 * E15 * E18  # no decimals
X1 = X0 // 2  # 18 decimals
A1 = 6_235_149_080_811_616_882_910 * E6  # no decimals
X2 = X1 * 100 // 2
A2 = 7_896_296_018_268_069_516_100 * E12
X3 = X2 // 2
A3 = 888_611_052_050_787_263_676 * E6
X4 = X3 // 2
A4 = 298_095_798_704_172_827_474 * E3
X5 = X4 // 2
A5 = 5_459_815_003_314_423_907_810
X6 = X5 // 2
A6 = 738_905_609_893_065_022_723
X7 = X6 // 2
A7 = 271_828_182_845_904_523_536
X8 = X7 // 2
A8 = 164_872_127_070_012_814_685
X9 = X8 // 2
A9 = 128_402_541_668_774_148_407
X10 = X9 // 2
A10 = 11_331_4845_306_682_631_683
X11 = X10 // 2
A11 = 1_064_49_445_891_785_942_956


def unsafe_add(a, b):
    return a + b


def unsafe_sub(a, b):
    return a - b


def unsafe_mul(a, b):
    return a * b


def unsafe_div(a, b):
    return a // b


def convert(value, _type):
    return _type(value)


class LogExpMath:

    @staticmethod
    def _pow_up(_x: int, _y: int) -> int:
        p = LogExpMath._pow(_x, _y)
        if p == 0:
            return 0
        return unsafe_add(
            unsafe_add(
                p, unsafe_div(unsafe_sub(unsafe_mul(
                    p, MAX_POW_REL_ERR), 1), PRECISION)
            ),
            1,
        )

    @staticmethod
    def _pow_down(_x: int, _y: int) -> int:
        p = LogExpMath._pow(_x, _y)
        if p == 0:
            return 0
        e = unsafe_add(
            unsafe_div(unsafe_sub(unsafe_mul(
                p, MAX_POW_REL_ERR), 1), PRECISION), 1
        )
        if p < e:
            return 0
        return unsafe_sub(p, e)

    @staticmethod
    def _pow(_x: int, _y: int) -> int:
        if _y == 0:
            return E18  # x^0 == 1

        if _x == 0:
            return 0  # 0^y == 0

        assert (_x >> int(255)) == 0, "x out of bounds"
        assert _y < MILD_EXP_BOUND, "y out of bounds"

        x = int(_x)
        y = int(_y)
        l = 0
        if LOG36_LOWER < x < LOG36_UPPER:
            l = LogExpMath._log36(x)
            l = unsafe_add(
                unsafe_mul(unsafe_div(l, E18), y),
                unsafe_div(unsafe_mul(l % E18, y), E18),
            )
        else:
            l = unsafe_mul(LogExpMath._log(x), y)
        l = unsafe_div(l, E18)
        return convert(LogExpMath._exp(l), int)

    @staticmethod
    def _log36(_x: int) -> int:
        x = unsafe_mul(_x, E18)
        z = unsafe_div(unsafe_mul(unsafe_sub(x, E36), E36), unsafe_add(x, E36))
        zsq = unsafe_div(unsafe_mul(z, z), E36)
        n = z
        c = z

        n = unsafe_div(unsafe_mul(n, zsq), E36)
        c = unsafe_add(c, n // 3)
        n = unsafe_div(unsafe_mul(n, zsq), E36)
        c = unsafe_add(c, n // 5)
        n = unsafe_div(unsafe_mul(n, zsq), E36)
        c = unsafe_add(c, n // 7)
        n = unsafe_div(unsafe_mul(n, zsq), E36)
        c = unsafe_add(c, n // 9)
        n = unsafe_div(unsafe_mul(n, zsq), E36)
        c = unsafe_add(c, n // 11)
        n = unsafe_div(unsafe_mul(n, zsq), E36)
        c = unsafe_add(c, n // 13)
        n = unsafe_div(unsafe_mul(n, zsq), E36)
        c = unsafe_add(c, n // 15)

        return unsafe_mul(c, 2)

    @staticmethod
    def _log(_a: int) -> int:
        if _a < E18:
            return -LogExpMath.__log(unsafe_div(unsafe_mul(E18, E18), _a))
        return LogExpMath.__log(_a)

    @staticmethod
    def __log(_a: int) -> int:
        a = _a
        s = 0

        if a >= unsafe_mul(A0, E18):
            a = unsafe_div(a, A0)
            s = unsafe_add(s, X0)
        if a >= unsafe_mul(A1, E18):
            a = unsafe_div(a, A1)
            s = unsafe_add(s, X1)

        a = unsafe_mul(a, 100)
        s = unsafe_mul(s, 100)

        if a >= A2:
            a = unsafe_div(unsafe_mul(a, E20), A2)
            s = unsafe_add(s, X2)
        if a >= A3:
            a = unsafe_div(unsafe_mul(a, E20), A3)
            s = unsafe_add(s, X3)
        if a >= A4:
            a = unsafe_div(unsafe_mul(a, E20), A4)
            s = unsafe_add(s, X4)
        if a >= A5:
            a = unsafe_div(unsafe_mul(a, E20), A5)
            s = unsafe_add(s, X5)
        if a >= A6:
            a = unsafe_div(unsafe_mul(a, E20), A6)
            s = unsafe_add(s, X6)
        if a >= A7:
            a = unsafe_div(unsafe_mul(a, E20), A7)
            s = unsafe_add(s, X7)
        if a >= A8:
            a = unsafe_div(unsafe_mul(a, E20), A8)
            s = unsafe_add(s, X8)
        if a >= A9:
            a = unsafe_div(unsafe_mul(a, E20), A9)
            s = unsafe_add(s, X9)
        if a >= A10:
            a = unsafe_div(unsafe_mul(a, E20), A10)
            s = unsafe_add(s, X10)
        if a >= A11:
            a = unsafe_div(unsafe_mul(a, E20), A11)
            s = unsafe_add(s, X11)

        z = unsafe_div(unsafe_mul(unsafe_sub(a, E20), E20), unsafe_add(a, E20))
        zsq = unsafe_div(unsafe_mul(z, z), E20)
        n = z
        c = z

        n = unsafe_div(unsafe_mul(n, zsq), E20)
        c = unsafe_add(c, unsafe_div(n, 3))
        n = unsafe_div(unsafe_mul(n, zsq), E20)
        c = unsafe_add(c, unsafe_div(n, 5))
        n = unsafe_div(unsafe_mul(n, zsq), E20)
        c = unsafe_add(c, unsafe_div(n, 7))
        n = unsafe_div(unsafe_mul(n, zsq), E20)
        c = unsafe_add(c, unsafe_div(n, 9))
        n = unsafe_div(unsafe_mul(n, zsq), E20)
        c = unsafe_add(c, unsafe_div(n, 11))

        c = unsafe_mul(c, 2)
        return unsafe_div(unsafe_add(s, c), 100)

    @staticmethod
    def _exp(_x: int) -> int:
        assert MIN_NAT_EXP <= _x <= MAX_NAT_EXP, "exp out of bounds"
        if _x < 0:
            return unsafe_mul(E18, E18) // LogExpMath.__exp(-_x)
        return LogExpMath.__exp(_x)

    @staticmethod
    def __exp(_x: int) -> int:
        x = _x
        f = 1
        if x >= X0:
            x = unsafe_sub(x, X0)
            f = A0
        elif x >= X1:
            x = unsafe_sub(x, X1)
            f = A1

        x = unsafe_mul(x, 100)
        p = E20

        if x >= X2:
            x = unsafe_sub(x, X2)
            p = unsafe_div(unsafe_mul(p, A2), E20)
        if x >= X3:
            x = unsafe_sub(x, X3)
            p = unsafe_div(unsafe_mul(p, A3), E20)
        if x >= X4:
            x = unsafe_sub(x, X4)
            p = unsafe_div(unsafe_mul(p, A4), E20)
        if x >= X5:
            x = unsafe_sub(x, X5)
            p = unsafe_div(unsafe_mul(p, A5), E20)
        if x >= X6:
            x = unsafe_sub(x, X6)
            p = unsafe_div(unsafe_mul(p, A6), E20)
        if x >= X7:
            x = unsafe_sub(x, X7)
            p = unsafe_div(unsafe_mul(p, A7), E20)
        if x >= X8:
            x = unsafe_sub(x, X8)
            p = unsafe_div(unsafe_mul(p, A8), E20)
        if x >= X9:
            x = unsafe_sub(x, X9)
            p = unsafe_div(unsafe_mul(p, A9), E20)

        n = x
        c = unsafe_add(E20, x)

        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 2)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 3)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 4)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 5)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 6)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 7)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 8)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 9)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 10)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 11)
        c = unsafe_add(c, n)
        n = unsafe_div(unsafe_div(unsafe_mul(n, x), E20), 12)
        c = unsafe_add(c, n)

        return unsafe_div(unsafe_mul(unsafe_div(unsafe_mul(p, c), E20), f), 100)


#  instantiate math lib
math = LogExpMath()


@profile
async def get_output_token(_i: int, _j: int, _dx: int) -> int:
    #  num_tokens = pool.functions.numTokens().call()
    num_tokens = 4
    assert _i != _j  # dev: same input and output asset
    assert _i < num_tokens and _j < num_tokens  # dev: index out of bounds
    assert _dx > 0  # dev: zero amount

    # update rates for from and to assets
    supply = 0
    amplification = 0
    vb_prod = 0
    vb_sum = 0
    vb_prod, vb_sum = await pool.functions.virtualBalanceProdSum().call()
    packed_weights = []
    rates = []
    supply, amplification, vb_prod, vb_sum, packed_weights, rates = await _get_rates(
        unsafe_add(_i, 1) | (unsafe_add(_j, 1) << 8), vb_prod, vb_sum
    )
    prev_vb_sum = vb_sum

    prev_vb_x = (
        await pool.functions.virtualBalance(_i).call()
        * rates[0]
        / (await pool.functions.rate(_i).call())
    )
    wn_x = _unpack_wn(packed_weights[_i], num_tokens)

    prev_vb_y = (
        await pool.functions.virtualBalance(_j).call()
        * rates[1]
        / (await pool.functions.rate(_j).call())
    )
    wn_y = _unpack_wn(packed_weights[_j], num_tokens)

    dx_fee = _dx * pool.functions.swapFeeRate().call() / PRECISION
    dvb_x = (_dx - dx_fee) * rates[0] / PRECISION
    vb_x = prev_vb_x + dvb_x

    # update x_i and remove x_j from variables
    vb_prod = (
        vb_prod
        * math._pow_up(int(prev_vb_y), (wn_y))
        / math._pow_down(int(vb_x * PRECISION // prev_vb_x), int(wn_x))
    )
    vb_sum = vb_sum + dvb_x - prev_vb_y

    # calulate new balance of out token
    vb_y = _calc_vb(
        int(wn_y),
        int(prev_vb_y),
        int(supply),
        int(amplification),
        int(vb_prod),
        int(vb_sum),
    )
    vb_sum += vb_y + dx_fee * rates[0] // PRECISION

    # check bands
    _check_bands(
        prev_vb_x * PRECISION // prev_vb_sum,
        vb_x * PRECISION // vb_sum,
        packed_weights[_i],
    )
    _check_bands(
        prev_vb_y * PRECISION // prev_vb_sum,
        vb_y * PRECISION // vb_sum,
        packed_weights[_j],
    )

    return (prev_vb_y - vb_y) * PRECISION / rates[1]


@profile
async def get_add_lp(_amounts: List[int]) -> int:
    batch = web3.batch_requests()

    rates = []
    virtual_balances = []
    packed_weights = []
    weights = []

    print("preparing batch")

    [batch.add(pool.functions.rate(t).call()) for t in range(4)]
    [batch.add(pool.functions.virtualBalance(t).call()) for t in range(4)]
    [batch.add(pool.functions.packedWeight(t).call()) for t in range(4)]
    [batch.add(pool.functions.weight(t).call()) for t in range(4)]

    batch.add(pool.functions.supply().call())
    batch.add(pool.functions.virtualBalanceProdSum().call())
    batch.add(pool.functions.rampLastTime().call())
    batch.add(pool.functions.rampStopTime().call())
    batch.add(pool.functions.rampStep().call())
    batch.add(pool.functions.amplification().call())
    batch.add(pool.functions.targetAmplification().call())
    batch.add(web3.eth.get_block("latest"))

    responses = await batch.async_execute()

    rates = responses[:4]
    virtual_balances = responses[4:8]
    packed_weights = responses[8:12]
    weights = responses[12:16]
    supply = responses[16]
    vb_prod, vb_sum = responses[17]
    span = responses[18]
    duration = responses[19]
    ramp_step = responses[20]
    amplification = responses[21]
    target_amplification = responses[22]
    timestamp = responses[23]["timestamp"]

    print(f"Rates: {rates}")
    print(f"Virtual Balances: {virtual_balances}")
    print(f"Packed Weights: {packed_weights}")
    print(f"Weights: {weights}")
    print(f"Supply: {supply}")
    print(f"Virtual Balance Product: {vb_prod}")
    print(f"Virtual Balance Sum: {vb_sum}")
    print(f"Span: {span}")
    print(f"Duration: {duration}")
    print(f"Ramp Step: {ramp_step}")
    print(f"Amplification: {amplification}")
    print(f"Target Amplification: {target_amplification}")
    print(f"Timestamp: {timestamp}")

    num_tokens = 4
    assert len(_amounts) == num_tokens
    assert vb_sum > 0

    # find lowest relative increase in balance
    tokens = 0
    lowest = 2**256 - 1
    sh = 0
    for token in range(MAX_NUM_ASSETS):
        if token == num_tokens:
            break
        if _amounts[token] > 0:
            tokens = tokens | (unsafe_add(token, 1) << sh)
            sh = unsafe_add(sh, 8)
            if vb_sum > 0 and lowest > 0:
                lowest = min(
                    _amounts[token] *
                    rates[token] // virtual_balances[token], lowest
                )
        else:
            lowest = 0
    assert sh > 0

    # update rates
    prev_supply = 0
    amplification = 0
    packed_weights = []
    rates = []

    prev_supply, amplification, vb_prod, vb_sum, packed_weights, rates = (
        await _get_rates(
            tokens,
            weights,
            supply,
            virtual_balances,
            vb_prod,
            vb_sum,
            span,
            duration,
            timestamp,
            ramp_step,
            amplification,
            target_amplification,
        )
    )

    vb_prod_final = vb_prod
    vb_sum_final = vb_sum
    # fee_rate = int((await pool.functions.swapFeeRate().call()) / 2)
    fee_rate = int(150000000000000)
    prev_vb_sum = vb_sum
    balances = []
    j = 0
    for token in range(MAX_NUM_ASSETS):
        if token == num_tokens:
            break

        amount = _amounts[token]
        if amount == 0:
            continue

        prev_vb = (
            (await pool.functions.virtualBalance(token).call())
            * rates[j]
            / (await pool.functions.rate(token).call())
        )

        dvb = amount * rates[j] / PRECISION
        vb = prev_vb + dvb
        balances.append(vb)

        if prev_supply > 0:
            wn = _unpack_wn(packed_weights[token], num_tokens)

            # update product and sum of virtual balances
            vb_prod_final = (
                vb_prod_final
                * math._pow_up(int(prev_vb * PRECISION / vb), wn)
                / PRECISION
            )
            # the `D^n` factor will be updated in `_calc_supply()`
            vb_sum_final += dvb

            # remove fees from balance and recalculate sum and product
            fee = (dvb - prev_vb * lowest / PRECISION) * fee_rate / PRECISION
            vb_prod = (
                vb_prod
                * math._pow_up(int(prev_vb * PRECISION / (vb - fee)), wn)
                / PRECISION
            )
            vb_sum += dvb - fee
        j = unsafe_add(j, 1)

    #  check bands
    j = 0
    for token in range(MAX_NUM_ASSETS):
        if token == num_tokens:
            break
        if _amounts[token] == 0:
            continue
        _check_bands(
            (await pool.functions.virtualBalance(token).call())
            * rates[j]
            / (await pool.functions.rate(token).call())
            * PRECISION
            / prev_vb_sum,
            balances[j] * PRECISION / vb_sum_final,
            packed_weights[token],
        )
        j = unsafe_add(j, 1)

    supply = 0
    (supply, vb_prod) = _calc_supply(
        int(num_tokens),
        int(prev_supply),
        int(amplification),
        int(vb_prod),
        int(vb_sum),
        prev_supply == 0,
    )
    return supply - prev_supply


@profile
async def get_remove_lp(_lp_amount: int) -> List[int]:
    amounts = []
    num_tokens = 4

    rates = []
    prev_balances = []

    batch = web3.batch_requests()

    batch.add(pool.functions.supply().call())

    for t in range(4):
        batch.add(pool.functions.virtualBalance(t).call())
        batch.add(pool.functions.rate(t).call())

    responses = await batch.async_execute()

    prev_supply = responses[0]

    for i in range(4):
        prev_balances.append(responses[2 * i + 1])
        rates.append(responses[2 * i + 2])

    assert _lp_amount <= prev_supply

    for token in range(MAX_NUM_ASSETS):
        if token == num_tokens:
            break
        prev_bal = prev_balances[token]
        dbal = prev_bal * _lp_amount // prev_supply
        amount = dbal * PRECISION // rates[token]
        amounts.append(amount)

    return amounts


@profile
def get_remove_single_lp(_token: int, _lp_amount: int) -> int:
    num_tokens = pool.functions.numTokens().call()
    assert _token < num_tokens

    #  update rate
    prev_supply = 0
    amplification = 0
    vb_prod = 0
    vb_sum = 0
    vb_prod, vb_sum = pool.functions.virtualBalanceProdSum().call()
    packed_weights = []
    rates = []
    prev_supply, amplification, vb_prod, vb_sum, packed_weights, rates = _get_rates(
        unsafe_add(_token, 1), vb_prod, vb_sum
    )
    prev_vb_sum = vb_sum

    supply = prev_supply - _lp_amount
    prev_vb = (
        pool.functions.virtualBalance(_token).call()
        * rates[0]
        / pool.functions.rate(_token).call()
    )
    wn = _unpack_wn(packed_weights[_token], num_tokens)

    #  update variables
    vb_prod = vb_prod * math._pow_up(int(prev_vb), int(wn)) / PRECISION
    for i in range(MAX_NUM_ASSETS):
        if i == num_tokens:
            break
        vb_prod = vb_prod * supply / prev_supply
    vb_sum = vb_sum - prev_vb

    #  calculate new balance of token
    vb = _calc_vb(
        int(wn),
        int(prev_vb),
        int(supply),
        int(amplification),
        int(vb_prod),
        int(vb_sum),
    )
    dvb = prev_vb - vb
    fee = int(dvb) * pool.functions.swapFeeRate().call() // 2 // PRECISION
    dvb -= fee
    vb += fee
    dx = dvb * PRECISION / rates[0]
    vb_sum = vb_sum + vb

    for token in range(MAX_NUM_ASSETS):
        if token == num_tokens:
            break
        if token == _token:
            _check_bands(
                prev_vb * PRECISION // prev_vb_sum,
                vb * PRECISION // vb_sum,
                packed_weights[token],
            )
        else:
            bal = pool.functions.virtualBalance(token).call()
            _check_bands(
                bal * PRECISION / prev_vb_sum,
                bal * PRECISION // vb_sum,
                packed_weights[token],
            )

    return dx


@profile
async def _get_rates(
    _tokens: int,
    _weights: List[List[int]],
    _supply: int,
    _virtual_balances: List[int],
    _vb_prod: int,
    _vb_sum: int,
    span: int,
    duration: int,
    timestamp: int,
    ramp_step: int,
    amplification: int,
    target_amplification: int,
) -> (int, int, int, int, List[int], List[int]):
    packed_weights = []
    rates = []

    amplification = 0
    vb_prod = 0
    vb_sum = _vb_sum
    amplification, vb_prod, _packed_weights, updated = await _get_packed_weights(
        _weights,
        _supply,
        _virtual_balances,
        _vb_prod,
        _vb_sum,
        span,
        duration,
        timestamp,
        ramp_step,
        amplification,
        target_amplification,
    )

    # num_tokens = pool.functions.numTokens().call()
    num_tokens = 4

    if not updated:
        for token in range(MAX_NUM_ASSETS):
            if token == num_tokens:
                break
            packed_weight = await pool.functions.packedWeight(token).call()
            packed_weights.append(packed_weight)

    for i in range(MAX_NUM_ASSETS):
        token = (_tokens >> unsafe_mul(8, i)) & 255
        if token == 0 or token > num_tokens:
            break
        token = unsafe_sub(token, 1)
        prev_rate = await pool.functions.rate(token).call()

        provider_address = await pool.functions.rateProviders(token).call()
        provider = web3.eth.contract(
            address=provider_address, abi=rate_provider_abi)
        rate = await provider.functions.rate(
            await pool.functions.tokens(token).call()
        ).call()
        assert rate > 0
        rates.append(rate)

        if rate == prev_rate:
            continue

        if prev_rate > 0 and vb_sum > 0:
            # factor out old rate and factor in new
            wn = _unpack_wn(packed_weights[token], num_tokens)
            vb_prod = (
                vb_prod
                * math._pow_up(int(prev_rate * PRECISION // rate), wn)
                / PRECISION
            )

            prev_bal = await pool.functions.virtualBalance(token).call()
            bal = prev_bal * rate // prev_rate
            vb_sum = vb_sum + bal - prev_bal

    if not updated and vb_prod == _vb_prod and vb_sum == _vb_sum:
        return (
            (await pool.functions.supply().call()),
            amplification,
            vb_prod,
            vb_sum,
            packed_weights,
            rates,
        )

    supply = 0
    (supply, vb_prod) = _calc_supply(
        int(num_tokens),
        int(await pool.functions.supply().call()),
        int(amplification),
        int(vb_prod),
        int(vb_sum),
        True,
    )
    return supply, amplification, vb_prod, vb_sum, packed_weights, rates


@profile
async def _get_packed_weights(
    _weights: List[List[int]],
    _supply: int,
    _virtual_balances: List[int],
    _vb_prod: int,
    _vb_sum: int,
    _span: int,
    _duration: int,
    _timestamp: int,
    _ramp_step: int,
    _amplification: int,
    _target_amplification: int,
) -> (int, int, List[int], bool):
    packed_weights = []

    span = _span
    duration = _duration
    timestamp = _timestamp
    ramp_step = _ramp_step
    amplification = _amplification
    target_amplification = _target_amplification

    if (
        span == 0
        or span > timestamp
        or (timestamp - span < ramp_step and duration > timestamp)
    ):
        return (
            amplification,
            _vb_prod,
            packed_weights,
            False,
        )

    if timestamp < duration:
        # ramp in progress
        duration -= span
    else:
        #  ramp has finished
        duration = 0
    span = timestamp - span

    # update amplification
    current = amplification
    target = target_amplification

    if duration == 0:
        current = target
    else:
        if current > target:
            current = current - (current - target) * span / duration
        else:
            current = current + (target - current) * span / duration
    amplification = current

    #  update weights
    num_tokens = 4
    supply = _supply
    vb_prod = 0
    if _vb_sum > 0:
        vb_prod = PRECISION
    lower = 0
    upper = 0
    for token in range(MAX_NUM_ASSETS):
        if token == num_tokens:
            break
        current, target, lower, upper = _weights[token]
        if duration == 0:
            current = target
        else:
            if current > target:
                current -= (current - target) * span / duration
            else:
                current += (target - current) * span / duration
        packed_weights.append(_pack_weight(current, target, lower, upper))
        if _vb_sum > 0:
            vb_prod = unsafe_div(
                unsafe_mul(
                    vb_prod,
                    math._pow_down(
                        unsafe_div(
                            unsafe_mul(supply, current),
                            (await pool.functions.virtualBalance(token)),
                        ),
                        unsafe_mul(current, num_tokens),
                    ),
                ),
                PRECISION,
            )

    return amplification, vb_prod, packed_weights, True


@profile
def _pack_weight(_weight: int, _target: int, _lower: int, _upper: int) -> int:
    return (
        unsafe_div(_weight, WEIGHT_SCALE)
        | (unsafe_div(_target, WEIGHT_SCALE) << -TARGET_WEIGHT_SHIFT)
        | (unsafe_div(_lower, WEIGHT_SCALE) << -LOWER_BAND_SHIFT)
        | (unsafe_div(_upper, WEIGHT_SCALE) << -UPPER_BAND_SHIFT)
    )


@profile
def _unpack_weights(_packed: int) -> (int, int, int, int):
    return (
        unsafe_mul(_packed & WEIGHT_MASK, WEIGHT_SCALE),
        unsafe_mul((_packed >> -TARGET_WEIGHT_SHIFT)
                   & WEIGHT_MASK, WEIGHT_SCALE),
        unsafe_mul((_packed >> -LOWER_BAND_SHIFT) & WEIGHT_MASK, WEIGHT_SCALE),
        unsafe_mul((_packed >> -UPPER_BAND_SHIFT), WEIGHT_SCALE),
    )


@profile
def _unpack_wn(_packed: int, _num_tokens: int) -> int:
    return unsafe_mul(unsafe_mul(_packed & WEIGHT_MASK, WEIGHT_SCALE), _num_tokens)


@profile
def _calc_supply(
    _num_tokens: int,
    _supply: int,
    _amplification: int,
    _vb_prod: int,
    _vb_sum: int,
    _up: bool,
) -> (int, int):
    # s[n+1] = (A sum / w^n - s^(n+1) w^n /prod^n)) / (A w^n - 1)
    #        = (l - s r) / d

    l = _amplification
    d = l - PRECISION
    s = _supply
    r = _vb_prod
    l = l * _vb_sum

    num_tokens = _num_tokens
    for _ in range(255):
        sp = unsafe_div(unsafe_sub(l, unsafe_mul(s, r)), d)  # (l - s * r) / d
        for i in range(MAX_NUM_ASSETS):
            if i == num_tokens:
                break
            r = unsafe_div(unsafe_mul(r, sp), s)  # r * sp / s
        if sp >= s:
            if (sp - s) * PRECISION / s <= MAX_POW_REL_ERR:
                if _up:
                    sp += sp * MAX_POW_REL_ERR / PRECISION
                else:
                    sp -= sp * MAX_POW_REL_ERR / PRECISION
                return sp, r
        else:
            if (s - sp) * PRECISION / s <= MAX_POW_REL_ERR:
                if _up:
                    sp += sp * MAX_POW_REL_ERR / PRECISION
                else:
                    sp -= sp * MAX_POW_REL_ERR / PRECISION
                return sp, r
        s = sp

    raise "no convergence"


@profile
def _calc_vb(_wn, _y, _supply, _amplification, _vb_prod, _vb_sum) -> int:
    # y = x_j, sum' = sum(x_i, i != j), prod' = prod(x_i^w_i, i != j)
    # w = product(w_i), v_i = w_i n, f_i = 1/v_i
    # Iteratively find root of g(y) using Newton's method
    # g(y) = y^(v_j + 1) + (sum' + (w^n / A - 1) D y^(w_j n) - D^(n+1) w^2n / prod'^n
    #      = y^(v_j + 1) + b y^(v_j) - c
    # y[n+1] = y[n] - g(y[n])/g'(y[n])
    #        = (y[n]^2 + b (1 - f_j) y[n] + c f_j y[n]^(1 - v_j)) / (f_j + 1) y[n] + b)

    d = _supply
    b = d * PRECISION // _amplification  # actually b + D
    c = _vb_prod * b // PRECISION
    b += _vb_sum
    f = PRECISION * PRECISION // _wn

    y = _y
    for _ in range(255):
        yp = (
            (
                y
                + b
                + d * f // PRECISION
                + c * f // math._pow_up(int(y), int(_wn))
                - b * f // PRECISION
                - d
            )
            * y
            // (f * y // PRECISION + y + b - d)
        )
        if yp >= y:
            if (yp - y) * PRECISION // y <= MAX_POW_REL_ERR:
                yp += yp * MAX_POW_REL_ERR // PRECISION
                return yp
        else:
            if (y - yp) * PRECISION // y <= MAX_POW_REL_ERR:
                yp += yp * MAX_POW_REL_ERR // PRECISION
                return yp
        y = yp

    raise "no convergence"


@profile
def _check_bands(_prev_ratio, _ratio, _packed_weight):
    weight = unsafe_mul(_packed_weight & WEIGHT_MASK, WEIGHT_SCALE)

    # lower limit check
    limit = unsafe_mul(
        (_packed_weight >> -LOWER_BAND_SHIFT) & WEIGHT_MASK, WEIGHT_SCALE
    )
    if limit > weight:
        limit = 0
    else:
        limit = unsafe_sub(weight, limit)
    if _ratio < limit:
        assert _ratio > _prev_ratio  # dev: ratio below lower band

    # upper limit check
    limit = min(
        unsafe_add(
            weight, unsafe_mul(
                (_packed_weight >> -UPPER_BAND_SHIFT), WEIGHT_SCALE)
        ),
        PRECISION,
    )
    if _ratio > limit:
        assert _ratio < _prev_ratio  # dev: ratio above upper band


@profile
async def main():
    result = await get_add_lp(
        [
            int(0.1 * PRECISION),
            int(0.1 * PRECISION),
            int(0.02 * PRECISION),
            int(0.01 * PRECISION),
        ]
    )
    #  result = await get_remove_lp(1e6)
    print("result -", int(result))


if __name__ == "__main__":
    asyncio.run(main())

#  print(int(get_add_lp([int(0.1 * PRECISION), int(0.1 * PRECISION),
#       int(0.02 * PRECISION), int(0.01 * PRECISION)])))
# print(int(get_remove_single_lp(1, 1e8)))
# print(int(get_output_token(0, 1, 1e12)))
