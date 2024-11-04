import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

colnames = ['date','dblheader_code','day_of_week','team_v','league_v','game_no_v',
            'team_h','league_h','game_no_h', 'runs_v', 'runs_h','outs_total','day_night',
            'completion_info','forfeit_info','protest_info','ballpark_id','attendance','game_minutes',
            'linescore_v','linescore_h',
            'AB_v','H_v','2B_v','3B_v','HR_v','RBI_v','SH_v','SF_v','HBP_v','BB_v','IBB_v','SO_v',
            'SB_v', 'CS_v','GIDP_v','CI_v','LOB_v',
            'P_num_v','ERind_v','ERteam_v','WP_v','balk_v',
            'PO_v','ASST_v','ERR_v','PB_v','DP_v','TP_v',
            'AB_h', 'H_h', '2B_h', '3B_h', 'HR_h', 'RBI_h', 'SH_h', 'SF_h', 'HBP_h', 'BB_h', 'IBB_h','SO_h',
            'SB_h', 'CS_h', 'GIDP_h', 'CI_h', 'LOB_h',
            'P_num_h', 'ERind_h', 'ERteam_h', 'WP_h', 'balk_h',
            'PO_h', 'ASST_h', 'ERR_h', 'PB_h', 'DP_h', 'TP_h',
            'ump_HB_id', 'ump_HB_name','ump_1B_id', 'ump_1B_name','ump_2B_id', 'ump_2B_name',
            'ump_3B_id', 'ump_3B_name','ump_LF_id', 'ump_LF_name','ump_RF_id', 'ump_RF_name',
            'mgr_id_v', 'mgr_name_v', 'mgr_id_h', 'mgr_name_h',
            'pitcher_id_w','pitcher_name_w','pitcher_id_l','pitcher_name_l','pitcher_id_s','pitcher_name_s',
            'GWRBI_id','GWRBI_name','pitcher_start_id_v','pitcher_start_name_v','pitcher_start_id_h','pitcher_start_name_h',
            'batter1_name_v', 'batter1_id_v', 'batter1_pos_v', 'batter2_name_v', 'batter2_id_v', 'batter2_pos_v',
            'batter3_name_v', 'batter3_id_v', 'batter3_pos_v', 'batter4_name_v', 'batter4_id_v', 'batter4_pos_v',
            'batter5_name_v', 'batter5_id_v', 'batter5_pos_v', 'batter6_name_v', 'batter6_id_v', 'batter6_pos_v',
            'batter7_name_v', 'batter7_id_v', 'batter7_pos_v', 'batter8_name_v', 'batter8_id_v', 'batter8_pos_v',
            'batter9_name_v', 'batter9_id_v', 'batter9_pos_v', 'batter1_name_h', 'batter1_id_h', 'batter1_pos_h',
            'batter2_name_h', 'batter2_id_h', 'batter2_pos_h', 'batter3_name_h', 'batter3_id_h', 'batter3_pos_h',
            'batter4_name_h', 'batter4_id_h', 'batter4_pos_h', 'batter5_name_h', 'batter5_id_h', 'batter5_pos_h',
            'batter6_name_h', 'batter6_id_h', 'batter6_pos_h', 'batter7_name_h', 'batter7_id_h', 'batter7_pos_h',
            'batter8_name_h', 'batter8_id_h', 'batter8_pos_h', 'batter9_name_h', 'batter9_id_h', 'batter9_pos_h',
            'misc_info','acqui_info'
            ]


df = pd.DataFrame()
for year in range(1980,2024):
    df_temp = pd.read_csv(f'game_logs/gl{year}.txt', header=None)
    df_temp.columns = colnames
    df_temp['season'] = year
    df = pd.concat((df, df_temp))


df.reset_index(drop=True, inplace=True)

df['run_diff'] = df['runs_h'] - df['runs_v']
df['home_victory'] = (df['run_diff'] > 0).astype(int)
df['run_total'] = df['runs_h'].copy()+df['runs_v'].copy()
df['date_dblhead'] = (df['date'].astype(str) + df['dblheader_code'].astype(str)).astype(int)


def strip_suffix(x, suff):
    if x.endswith(suff):
        return x[:-len(suff)]
    else:
        return x


visit_cols = [col for col in df.columns if not col.endswith('_h')]
visit_cols_stripped = [strip_suffix(col, '_v') for col in visit_cols]
home_cols = [col for col in df.columns if not col.endswith('_v')]
home_cols_stripped = [strip_suffix(col, '_h') for col in home_cols]


def create_team_df(fteam):
    df_team_v = df[df.team_v == fteam]
    opponent = df_team_v['team_h']
    df_team_v = df_team_v[visit_cols]
    df_team_v.columns = visit_cols_stripped
    df_team_v['home_game'] = 0
    df_team_v['opponent'] = opponent

    df_team_h = df[df.team_h == fteam]
    opponent = df_team_h['team_v']
    df_team_h = df_team_h[home_cols]
    df_team_h.columns = home_cols_stripped
    df_team_h['home_game'] = 1
    df_team_h['opponent'] = opponent

    df_team = pd.concat((df_team_h, df_team_v))
    df_team.sort_values(['date', 'game_no'], inplace=True)

    for winsize in [162,30]:
        suff = str(winsize)
        for raw_col in ['AB', 'H', '2B', '3B', 'HR', 'BB', 'runs', 'SB', 'CS', 'ERR']:
            new_col = f'rollsum_{raw_col}_{suff}'
            df_team[new_col] = df_team[raw_col].rolling(winsize, closed='left').sum()

        df_team[f'rollsum_BATAVG_{suff}'] = df_team[f'rollsum_H_{suff}'] / df_team[f'rollsum_AB_{suff}']
        df_team[f'rollsum_OBP_{suff}'] = (df_team[f'rollsum_H_{suff}'] + df_team[f'rollsum_BB_{suff}']) / (
                                          df_team[f'rollsum_AB_{suff}'] + df_team[f'rollsum_BB_{suff}'])
        df_team[f'rollsum_SLG_{suff}'] = (df_team[f'rollsum_H_{suff}'] + df_team[f'rollsum_2B_{suff}'] +
                                          2*df_team[f'rollsum_3B_{suff}'] + 3*df_team[f'rollsum_3B_{suff}']) / (
                                          df_team[f'rollsum_AB_{suff}'])

        df_team[f'rollsum_OBS_{suff}'] = df_team[f'rollsum_OBP_{suff}'] + df_team[f'rollsum_SLG_{suff}']

    df_team['season_game'] = df_team['season']*1000 + df_team['game_no']
    df_team.set_index('season_game', inplace=True)
    return df_team


team_data_dict = {}
for team in df.team_v.unique():
    team_data_dict[team] = create_team_df(team)


BATAVG_162_h = np.zeros(df.shape[0])
BATAVG_162_v = np.zeros(df.shape[0])
OBP_162_h = np.zeros(df.shape[0])
OBP_162_v = np.zeros(df.shape[0])
SLG_162_h = np.zeros(df.shape[0])
SLG_162_v = np.zeros(df.shape[0])
OBS_162_h = np.zeros(df.shape[0])
OBS_162_v = np.zeros(df.shape[0])
SB_162_h = np.zeros(df.shape[0])
SB_162_v = np.zeros(df.shape[0])
CS_162_h = np.zeros(df.shape[0])
CS_162_v = np.zeros(df.shape[0])
ERR_162_h = np.zeros(df.shape[0])
ERR_162_v = np.zeros(df.shape[0])
BATAVG_30_h = np.zeros(df.shape[0])
BATAVG_30_v = np.zeros(df.shape[0])
OBP_30_h = np.zeros(df.shape[0])
OBP_30_v = np.zeros(df.shape[0])
SLG_30_h = np.zeros(df.shape[0])
SLG_30_v = np.zeros(df.shape[0])
OBS_30_h = np.zeros(df.shape[0])
OBS_30_v = np.zeros(df.shape[0])
SB_30_h = np.zeros(df.shape[0])
SB_30_v = np.zeros(df.shape[0])
CS_30_h = np.zeros(df.shape[0])
CS_30_v = np.zeros(df.shape[0])
ERR_30_h = np.zeros(df.shape[0])
ERR_30_v = np.zeros(df.shape[0])
i = 0
for index, row in df.iterrows():
    if i % 1000 == 0:
        print(i)
    home_team = row['team_h']
    visit_team = row['team_v']
    game_index_v = row['season']*1000 + row['game_no_v']
    game_index_h = row['season']*1000 + row['game_no_h']
    BATAVG_162_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_BATAVG_162']
    BATAVG_162_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_BATAVG_162']
    OBP_162_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_OBP_162']
    OBP_162_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_OBP_162']
    SLG_162_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_SLG_162']
    SLG_162_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_SLG_162']
    OBS_162_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_OBS_162']
    OBS_162_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_OBS_162']
    SB_162_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_SB_162']
    SB_162_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_SB_162']
    CS_162_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_CS_162']
    CS_162_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_CS_162']
    ERR_162_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_ERR_162']
    ERR_162_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_ERR_162']
    BATAVG_30_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_BATAVG_30']
    BATAVG_30_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_BATAVG_30']
    OBP_30_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_OBP_30']
    OBP_30_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_OBP_30']
    SLG_30_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_SLG_30']
    SLG_30_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_SLG_30']
    OBS_30_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_OBS_30']
    OBS_30_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_OBS_30']
    SB_30_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_SB_30']
    SB_30_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_SB_30']
    CS_30_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_CS_30']
    CS_30_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_CS_30']
    ERR_30_h[i] = team_data_dict[home_team].loc[game_index_h,'rollsum_ERR_30']
    ERR_30_v[i] = team_data_dict[visit_team].loc[game_index_v,'rollsum_ERR_30']
    i += 1


df['BATAVG_162_h'] = BATAVG_162_h
df['BATAVG_162_v'] = BATAVG_162_v
df['OBP_162_h'] = OBP_162_h
df['OBP_162_v'] = OBP_162_v
df['SLG_162_h'] = SLG_162_h
df['SLG_162_v'] = SLG_162_v
df['OBS_162_h'] = OBS_162_h
df['OBS_162_v'] = OBS_162_v
df['SB_162_h'] = SB_162_h
df['SB_162_v'] = SB_162_v
df['CS_162_h'] = CS_162_h
df['CS_162_v'] = CS_162_v
df['ERR_162_h'] = ERR_162_h
df['ERR_162_v'] = ERR_162_v
df['BATAVG_30_h'] = BATAVG_30_h
df['BATAVG_30_v'] = BATAVG_30_v
df['OBP_30_h'] = OBP_30_h
df['OBP_30_v'] = OBP_30_v
df['SLG_30_h'] = SLG_30_h
df['SLG_30_v'] = SLG_30_v
df['OBS_30_h'] = OBS_30_h
df['OBS_30_v'] = OBS_30_v
df['SB_30_h'] = SB_30_h
df['SB_30_v'] = SB_30_v
df['CS_30_h'] = CS_30_h
df['CS_30_v'] = CS_30_v
df['ERR_30_h'] = ERR_30_h
df['ERR_30_v'] = ERR_30_v

df.to_csv('df_bp.csv', index=False)