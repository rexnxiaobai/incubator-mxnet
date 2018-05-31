import mxnet as mx
from mxnet.test_utils import rand_ndarray, assert_almost_equal
import numpy as np
import time
import scipy

# test the bug of dot(csr.T, dns)=dns on gpu
def check_dot_determinism(lhs_stype, rhs_stype, lhs_density, rhs_density
    , transpose_a, transpose_b, dev, forward_stype='default'):
    lhs_shape = (200, 200)
    rhs_shape = (200, 200)
    # lhs_shape = (3000, 3000)
    # rhs_shape = (3000, 3000)
    lhs = rand_ndarray(lhs_shape, lhs_stype, density=lhs_density).as_in_context(dev)
    rhs = rand_ndarray(rhs_shape, rhs_stype, density=rhs_density).as_in_context(dev)
    res1 = mx.nd.dot(lhs, rhs, transpose_a=transpose_a, transpose_b=transpose_b
        , forward_stype=forward_stype)
    res2 = mx.nd.dot(lhs, rhs, transpose_a=transpose_a, transpose_b=transpose_b
        , forward_stype=forward_stype)

    assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=0.0, atol=0.0)

def measure_cost(repeat, f, *args, **kwargs):
    # start bench
    start = time.time()
    results = []
    for i in range(repeat):
        results.append(f(*args, **kwargs))
    for result in results:
        result.wait_to_read()
    end = time.time()
    diff = end - start
    return diff / repeat

def measure_fallback(repeat, a):
    # start bench
    start = time.time()
    results = []
    for i in range(repeat):
        results.append(a.tostype('default'))
    for result in results:
        result.wait_to_read()
    end = time.time()
    diff = end - start
    return diff / repeat

def create_data(shape, dev, stype='default', density=None):
    if density is not None:
        csr = scipy.sparse.random(shape[0], shape[1], density=density, format = 'csr', dtype=np.float32)
        mx_sparse = mx.nd.sparse.csr_matrix((csr.data, csr.indices, csr.indptr), shape=shape, ctx=dev)
        if stype=='default':
            mx_sparse = mx_sparse.tostype('default')
        elif stype == 'row_sparse':
            mx_sparse = mx_sparse.tostype('default').tostype('row_sparse')
        return mx_sparse
    else:
        dns = np.random.uniform(size=shape)
        mx_dns = mx.nd.array(dns, ctx=dev)
        return mx_dns

def testCorrectnessAndPerformance(lhs, rhs, lhs_transpose=False, rhs_transpose=False
    , forward_stype='default', repeat_num=50):
    # check correctness
    check = mx.nd.dot(lhs, rhs, transpose_a=lhs_transpose
                      , transpose_b=rhs_transpose, forward_stype=forward_stype)
    if lhs.stype != 'default':
        lhs_dns = lhs.tostype('default')
    else:
        lhs_dns = lhs
    if rhs.stype != 'default':
        rhs_dns = rhs.tostype('default')
    else:
        rhs_dns = rhs
    if lhs_transpose is False and rhs_transpose is False:
        check_np = np.dot(lhs_dns.asnumpy(), rhs_dns.asnumpy())
    elif lhs_transpose and rhs_transpose is False:
        check_np = np.dot(lhs_dns.asnumpy().T, rhs_dns.asnumpy())
    elif lhs_transpose is False and rhs_transpose:
        check_np = np.dot(lhs_dns.asnumpy(), rhs_dns.asnumpy().T)
    elif lhs_transpose and rhs_transpose:
        check_np = np.dot(lhs_dns.asnumpy().T, rhs_dns.asnumpy().T)
    assert_almost_equal(check.asnumpy(), check_np, atol=1e-5, rtol=1e-4)

    # check_dns = mx.nd.dot(lhs_dns, rhs_dns, transpose_a=lhs_transpose, transpose_b=rhs_transpose)
    # assert_almost_equal(check.asnumpy(), check_dns.asnumpy(), atol=1e-5, rtol=1e-4)

    # test speed up
    sparse_cost = 0.0
    dns_cost = 0.0
    fallback_cost = 0.0
    mx.nd.waitall()
    for i in range(repeat_num):
        sparse_cost += measure_cost(1, mx.nd.dot, lhs, rhs, transpose_a=lhs_transpose
                      , transpose_b=rhs_transpose, forward_stype=forward_stype)
        if lhs.stype != 'default':
            fallback_cost += measure_fallback(1, lhs)
        if rhs.stype != 'default':
            fallback_cost += measure_fallback(1, rhs)
        dns_cost += measure_cost(1, mx.nd.dot, lhs_dns, rhs_dns
                                 , transpose_a=lhs_transpose, transpose_b=rhs_transpose)
    return sparse_cost, dns_cost, fallback_cost

# test dot(dns1, csr) = dns2
def test_dns_csr_dns_score(shape_lhs, shape_rhs, dev, repeat_num=50):
    print('---test dot(dns1, csr) = dns2, shape_lhs=(%d,%d), shape_rhs=(%d,%d)---'
          %(shape_lhs[0], shape_lhs[1], shape_rhs[0], shape_rhs[1]))
    lhs = create_data(shape_lhs, dev=dev)
    for density in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
        rhs = create_data(shape_rhs, dev=dev, stype='csr', density=density)
        sparse_cost, dns_cost, fallback_cost = testCorrectnessAndPerformance(
            lhs, rhs, forward_stype='default',repeat_num=repeat_num)
        print("%.2f %% with fallback: %.6f, without fallback: %.6f"% (density * 100,
             (dns_cost + fallback_cost) / sparse_cost, dns_cost / sparse_cost))
    mx.nd.waitall()

# test dot(dns1, csr.T) = dns2
def test_dns_csrT_dns_score(shape_lhs, shape_rhs, dev, repeat_num=50):
    print('---test dot(dns1, csr.T) = dns2, shape_lhs=(%d,%d), shape_rhs=(%d,%d)---'
          %(shape_lhs[0], shape_lhs[1], shape_rhs[0], shape_rhs[1]))
    lhs = create_data(shape_lhs, dev=dev)
    for density in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
        rhs = create_data(shape_rhs, dev=dev, stype='csr', density=density)
        sparse_cost, dns_cost, fallback_cost = testCorrectnessAndPerformance(
            lhs, rhs, rhs_transpose=True, forward_stype='default', repeat_num=repeat_num)
        print("%.2f %% with fallback: %.6f, without fallback: %.6f"% (density * 100,
             (dns_cost + fallback_cost) / sparse_cost, dns_cost / sparse_cost))
    mx.nd.waitall()

# test dot(csr.T, dns1) = dns2
def test_csrT_dns_dns_score(shape_lhs, shape_rhs, dev, repeat_num=50):
    print('---test dot(csr.T, dns1) = dns2, shape_lhs=(%d,%d), shape_rhs=(%d,%d)---'
          %(shape_lhs[0], shape_lhs[1], shape_rhs[0], shape_rhs[1]))
    rhs = create_data(shape_rhs, dev=dev)
    for density in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
        lhs = create_data(shape_lhs, dev=dev, stype='csr', density=density)
        sparse_cost, dns_cost, fallback_cost = testCorrectnessAndPerformance(
            lhs, rhs, lhs_transpose=True, forward_stype='default', repeat_num=repeat_num)
        print("%.2f %% with fallback: %.6f, without fallback: %.6f"% (density * 100,
             (dns_cost + fallback_cost) / sparse_cost, dns_cost / sparse_cost))
    mx.nd.waitall()


if __name__ == "__main__":
    # check_dot_determinism('csr', 'default', 0.5, 1.0, True, False, dev=mx.cpu(), forward_stype='default')
    # print('check dot determinism is passed on cpu')
    # check_dot_determinism('csr', 'default', 0.5, 1.0, True, False, dev=mx.gpu(), forward_stype='default')
    # print('check dot determinism is passed on gpu')
    
    shape_lhs = (256, 30000)
    shape_rhs = (30000, 30000)
    test_dns_csr_dns_score(shape_lhs, shape_rhs, mx.cpu(), repeat_num=50)
    test_dns_csrT_dns_score(shape_lhs, shape_rhs, mx.cpu(), repeat_num=50)

    # shape_lhs = (3000, 3000)
    # shape_rhs = (3000, 3000)
    # test_csrT_dns_dns_score(shape_lhs, shape_rhs, mx.gpu(), repeat_num=50)


