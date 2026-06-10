# 論文描画用の図・データをまとめるための関数
from typing import Dict, Union
from pathlib import Path
from os import makedirs

import pickle

from constants import OUTDIR
from utils import get_fpth_in_timeseries, dir_to_condition
from monitor import (
    plt_stream_line,
    get_latest_fumarole_prop,
    calc_liq500_statistic,
    liq_sat_at500,
    calc_tempe_change_rate,
    plot_results,
    img2mov,
    plot_fumarole_props_foreach_tstep,
    plt_regional_timeseries,
    plot_sum_foreach_tstep
                     )
from params import PARAMS

# 準定常の最後・Unrestシナリオの最後・浸透率動的変化シナリオで浸透率に大きな変化がある時点・浸透率固定
quasi_steady_dirs = [
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

unrest_dirs = [
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_20000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_25000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_30000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_100000.0_v\unrest\900.0_0.0_35000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_15000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_20000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_25000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_30000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_v\unrest\900.0_0.0_35000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_100000.0_v\unrest\900.0_0.0_15000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_100000.0_v\unrest\900.0_0.0_20000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_100000.0_v\unrest\900.0_0.0_25000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_100000.0_v\unrest\900.0_0.0_30000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_100000.0_v\unrest\900.0_0.0_35000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_v\unrest\900.0_0.0_15000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_v\unrest\900.0_0.0_20000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_v\unrest\900.0_0.0_25000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_v\unrest\900.0_0.0_30000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_1000.0_10000.0_v\unrest\900.0_0.0_35000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_15000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_20000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_25000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_30000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_100000.0_v\unrest\900.0_0.0_35000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_15000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_20000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_25000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_30000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_v\unrest\900.0_0.0_35000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_100000.0_v\unrest\900.0_0.0_15000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_100000.0_v\unrest\900.0_0.0_20000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_100000.0_v\unrest\900.0_0.0_25000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_100000.0_v\unrest\900.0_0.0_30000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_100000.0_v\unrest\900.0_0.0_35000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_v\unrest\900.0_0.0_15000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_v\unrest\900.0_0.0_20000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_v\unrest\900.0_0.0_25000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_v\unrest\900.0_0.0_30000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_v\unrest\900.0_0.0_35000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_15000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_20000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_25000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_30000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_100000.0_v\unrest\900.0_0.1_35000.0_10.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_15000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_20000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_25000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_30000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10.0_v\unrest\900.0_0.1_35000.0_10.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_100000.0_v\unrest\900.0_0.1_15000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_100000.0_v\unrest\900.0_0.1_20000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_100000.0_v\unrest\900.0_0.1_25000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_100000.0_v\unrest\900.0_0.1_30000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_100000.0_v\unrest\900.0_0.1_35000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_v\unrest\900.0_0.1_15000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_v\unrest\900.0_0.1_20000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_v\unrest\900.0_0.1_25000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_v\unrest\900.0_0.1_30000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_1000.0_10000.0_v\unrest\900.0_0.1_35000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_100000.0_v\unrest\900.0_0.1_15000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_100000.0_v\unrest\900.0_0.1_20000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_100000.0_v\unrest\900.0_0.1_25000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_100000.0_v\unrest\900.0_0.1_30000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_100000.0_v\unrest\900.0_0.1_35000.0_10000.0_100000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_v\unrest\900.0_0.1_15000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_v\unrest\900.0_0.1_20000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_v\unrest\900.0_0.1_25000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_v\unrest\900.0_0.1_30000.0_10000.0_v_d",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_v\unrest\900.0_0.1_35000.0_10000.0_v_d",
        # # dynamic permeability
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_15000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_20000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_25000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_30000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        r"F:\tarumai2\900.0_0.0_1000.0_10.0_1.0_v\unrest\900.0_0.0_35000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_1.0_v\unrest\900.0_0.0_15000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_1.0_v\unrest\900.0_0.0_20000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_1.0_v\unrest\900.0_0.0_25000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_1.0_v\unrest\900.0_0.0_30000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10.0_1.0_v\unrest\900.0_0.0_35000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_1.0_v\unrest\900.0_0.0_15000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_1.0_v\unrest\900.0_0.0_20000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_1.0_v\unrest\900.0_0.0_25000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_1.0_v\unrest\900.0_0.0_30000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.0_10000.0_10000.0_1.0_v\unrest\900.0_0.0_35000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_15000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_20000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_25000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_30000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10.0_1.0_v\unrest\900.0_0.1_35000.0_10.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_1.0_v\unrest\900.0_0.1_15000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_1.0_v\unrest\900.0_0.1_20000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_1.0_v\unrest\900.0_0.1_25000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_1.0_v\unrest\900.0_0.1_30000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
        # r"F:\tarumai2\900.0_0.1_10000.0_10000.0_1.0_v\unrest\900.0_0.1_35000.0_10000.0_1.0_v_d_dyn100000.0_brit_pf2.7",
    ]

def plt_static_last():
    param_latest_fumarole: Dict[tuple, Dict[str, Dict[str, float]]] = {}
    param_liq500_statistic: Dict[tuple, Dict[str, float]] = {}
    for dirpth in quasi_steady_dirs:
        print(dirpth)
        dirpth = Path(dirpth)
        fpth_ls = get_fpth_in_timeseries(dirpth)
        plt_stream_line(fpth_ls[-1],
                        flux_props=["FLUXI#T",
                                    "FLUXJ#T",
                                    "FLUXK#T",
                                    ],
                        axis="Y",
                        )
        condition = dir_to_condition(dirpth)
        param = PARAMS(temp_src=condition["temp"],
               comp1t=condition["comp1t"],
               inj_rate=condition["inj_rate"],
               perm_vent=condition["perm"],
               cap_scale=condition["cap_scale"],
               permf_cap=condition["permf_cap"],
               vk=condition["vk"],
               disperse_magmasrc=condition["d"],
               db=condition["db"],
               pfail=condition["pfail"]
               )
        props = param_latest_fumarole.setdefault(param, {})
        for prop_name in ("TEMPC", "COMP1T", "FLUXK#E"):
            props.setdefault(prop_name,
                             get_latest_fumarole_prop(dirpth,
                                                      prop_name,
                                                      calc_average=True))
        sg500_dct = liq_sat_at500(fpth_ls[-1])
        param_liq500_statistic.setdefault(param, calc_liq500_statistic(sg500_dct))
        plot_results(fpth_ls[-1],
                     axis=("Y",),
                     prop_ls=["TEMPC", "SAT#GAS", "COMP1T"]
                     )

    # save as pickle object
    outdir = Path(OUTDIR)
    savedir = outdir.joinpath("summary").joinpath("quasi-stationary")
    makedirs(savedir, exist_ok=True)
    with open(savedir.joinpath("param_latest_fumarole.pkl"), "wb") as pkf:
        pickle.dump(param_latest_fumarole, pkf, pickle.HIGHEST_PROTOCOL)
    with open(savedir.joinpath("param_liq500_statistic.pkl"), "wb") as pkf:
        pickle.dump(param_liq500_statistic, pkf, pickle.HIGHEST_PROTOCOL)

    return param_latest_fumarole, param_liq500_statistic

def get_unrest_results() -> None:
    # TODO: remove comment out
    param_liq500_statistic: Dict[tuple, Dict[str, float]] = {}
    param_fumarole_times: Dict[tuple, Dict[str, Dict[Union[float, str], float]]] = {}
    for dirpth in unrest_dirs:
        print(dirpth)
        dirpth = Path(dirpth)
        fpth_ls = get_fpth_in_timeseries(dirpth)
        # plt_stream_line(fpth_ls[-1],
        #                 flux_props=["FLUXI#T",
        #                             "FLUXJ#T",
        #                             "FLUXK#T",
        #                             ],
        #                 axis="Y",
        #                 )
        # sg500_dct = liq_sat_at500(fpth_ls[-1])
        condition = dir_to_condition(dirpth)
        param = PARAMS(temp_src=condition["temp"],
                       comp1t=condition["comp1t"],
                       inj_rate=condition["inj_rate"],
                       perm_vent=condition["perm"],
                       cap_scale=condition["cap_scale"],
                       permf_cap=condition["permf_cap"],
                       vk=condition["vk"],
                       disperse_magmasrc=condition["d"],
                       db=condition["db"],
                       pfail=condition["pfail"]
                       )
        # param_liq500_statistic.setdefault(param, calc_liq500_statistic(sg500_dct))
        fumarole_times = calc_tempe_change_rate(dirpth, criteria=(300.0, 500.0))
        param_fumarole_times.setdefault(param, fumarole_times)
    
    # save as pickle object
    outdir = Path(OUTDIR)
    savedir = outdir.joinpath("summary").joinpath("unrest")
    makedirs(savedir, exist_ok=True)
    with open(savedir.joinpath("param_fumarole_times.pkl"), "wb") as pkf:
        pickle.dump(param_fumarole_times, pkf, pickle.HIGHEST_PROTOCOL)

    # with open(savedir.joinpath("param_liq500_statistic.pkl"), "wb") as pkf:
    #     pickle.dump(param_liq500_statistic, pkf, pickle.HIGHEST_PROTOCOL)

    return param_fumarole_times, param_liq500_statistic


def plt_unrest_results():
    for dirpth in unrest_dirs:
        print(dirpth)
        # plot_sum_foreach_tstep(dirpth,
        #                        axes=("Y",),
        #                        prop_names=["TEMPC", "SAT#GAS", "PRES"],
        #                        idx_ls=([20,],))
        # plt_regional_timeseries(dirpth,)
        plot_fumarole_props_foreach_tstep(dirpth)
        # img2mov(Path(dirpth).joinpath("tstep").joinpath("PRES").joinpath("Y"))
        # img2mov(Path(dirpth).joinpath("tstep").joinpath("TEMPC").joinpath("Y"))
        # img2mov(Path(dirpth).joinpath("tstep").joinpath("SAT#GAS").joinpath("Y"))
        # 地盤変動・全磁力の時間変化



if __name__ == "__main__":
    # plt_static_last()
    # get_unrest_results()
    plt_unrest_results()
    pass