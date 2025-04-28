# @Author  : ChaoQiezi
# @Time    : 2025/4/11 下午2:23
# @Email   : chaoqiezi.one@qq.com
# @FileName: dead_code

"""
This script is used to
"""
def reClass(a,b,c,d,e,f,g,h,i,j,k,l,n,o,m,s):
    if(1.0*m/s<0.5):
        return "混合功能区"
    else:
        if(m == a):
            return "商业-绿地"
        if(m == b):
            return "商业-居住"
        if(m == c):
            return "商业-交通"
        if(m == d):
            return "商业-公共"
        if(m == e):
            return "商业-公业"
        if(m == f):
            return "绿地-居住"
        if(m == g):
            return "绿地-交通"
        if(m == h):
            return "绿地-公共"
        if(m == i):
          return "绿地-工业"
        if(m == j):
            return "居住-交通"
        if(m == k):
            return "居住-公共"
        if(m == l):
            return "居住-工业"
        if(m == n):
             return "交通-公共"
        if(m == o):
            return "交通-工业"
        else:
             return "公共-工业"

reClass(!sy_count! * !sy_b_mj! + !ld_count! * !ld_b_mj!, !sy_count! * !sy_b_mj! + !jz_count! * !jz_b_mj!, !sy_count! * !sy_b_mj! + !jt_count! * !jt_b_mj!, !sy_count! * !sy_b_mj! + !gg_count! * !gg_b_mj!, !sy_count! * !sy_b_mj! + !gy_count! * !gy_b_mj!, !ld_count! * !ld_b_mj! + !jz_count! * !jz_b_mj!, !ld_count! * !ld_b_mj! + !jt_count! * !jt_b_mj!, !ld_count! * !ld_b_mj! + !gg_count! * !gg_b_mj!, !ld_count! * !ld_b_mj! + !gy_count! * !gy_b_mj!, !jz_count! * !jz_b_mj! + !jt_count! * !jt_b_mj!, !jz_count! * !jz_b_mj! + !gg_count! * !gg_b_mj!, !jz_count! * !jz_b_mj! + !gy_count! * !gy_b_mj!, !jt_count! * !jt_b_mj! + !gg_count! * !gg_b_mj!, !jt_count! * !jt_b_mj! + !gy_count! * !gg_b_mj!, !gg_count! * !gg_b_mj! + !gy_count! * !gy_b_mj!, !max!, !sum! )
