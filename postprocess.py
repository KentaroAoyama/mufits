from pathlib import Path

from utils import get_fpth_in_timeseries
from monitor import plt_stream_line, get_latest_fumarole_prop, liq_sat_at500

# 準定常の最後・Unrestシナリオの最後・浸透率動的変化シナリオで浸透率に大きな変化がある時点・浸透率固定

def plt_static_last():
    dirpth_ls = [
        r"F:\tarumai2\900.0_0.0_100.0_10.0_1.0_v",
        r"F:\tarumai2\900.0_0.0_100.0_10.0_100000.0_v",
        r"F:\tarumai2\900.0_0.0_100.0_10.0_v",
        r"F:\tarumai2\900.0_0.0_100.0_10000.0_1.0_v",
        r"F:\tarumai2\900.0_0.0_100.0_10000.0_100000.0_v",
        r"F:\tarumai2\900.0_0.0_100.0_10000.0_v",
        r"F:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v",
        r"F:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v",
        r"F:\tarumai2\900.0_0.0_1000.0_10.0_v",
        r"F:\tarumai2\900.0_0.0_1000.0_10000.0_1.0_v",
        r"F:\tarumai2\900.0_0.0_1000.0_10000.0_100000.0_v",
        r"F:\tarumai2\900.0_0.0_1000.0_10000.0_v",
        r"F:\tarumai2\900.0_0.0_10000.0_10.0_1.0_v",
        r"F:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v",
        r"F:\tarumai2\900.0_0.0_10000.0_10.0_v",
        r"F:\tarumai2\900.0_0.0_10000.0_10000.0_1.0_v",
        r"F:\tarumai2\900.0_0.0_10000.0_10000.0_100000.0_v",
        r"F:\tarumai2\900.0_0.0_10000.0_10000.0_v",
        r"F:\tarumai2\900.0_0.1_100.0_10.0_1.0_v",
        r"F:\tarumai2\900.0_0.1_100.0_10.0_100000.0_v",
        r"F:\tarumai2\900.0_0.1_100.0_10.0_v",
        r"F:\tarumai2\900.0_0.1_100.0_10000.0_1.0_v",
        r"F:\tarumai2\900.0_0.1_100.0_10000.0_100000.0_v",
        r"F:\tarumai2\900.0_0.1_100.0_10000.0_v",
        r"F:\tarumai2\900.0_0.1_1000.0_10.0_1.0_v",
        r"F:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v",
        r"F:\tarumai2\900.0_0.1_1000.0_10.0_v",
        r"F:\tarumai2\900.0_0.1_1000.0_10000.0_1.0_v",
        r"F:\tarumai2\900.0_0.1_1000.0_10000.0_100000.0_v",
        r"F:\tarumai2\900.0_0.1_1000.0_10000.0_v",
        r"F:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v",
        r"F:\tarumai2\900.0_0.1_10000.0_10.0_100000.0_v",
        r"F:\tarumai2\900.0_0.1_10000.0_10.0_v",
        r"F:\tarumai2\900.0_0.1_10000.0_10000.0_1.0_v",
        r"F:\tarumai2\900.0_0.1_10000.0_10000.0_100000.0_v",
        r"F:\tarumai2\900.0_0.1_10000.0_10000.0_v"
                 ]
    for dirpth in dirpth_ls:
        fpth_ls = get_fpth_in_timeseries(dirpth)
        plt_stream_line(fpth_ls[-1])
        for prop_name in ("TEMPC", "COMP1T", "FLUXK#E"):
            get_latest_fumarole_prop(dirpth, prop_name, calc_average=True)
        liq_sat_at500(fpth_ls[-1])
    return


if __name__ == "__main__":
    plt_static_last()
    pass