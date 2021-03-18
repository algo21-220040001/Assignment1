import os
import re
import pandas as pd
import numpy as np
import collections
import functools
import pickle
import pdb
import scipy.stats as scst
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import Manager


class BacktestEngine:
    def __init__(self, backtest_mode, backtest_data, fac_data, benchmark,
                 num_groups):
        """
        回测模式 回测数据 因子文件 对冲标的 分组(用于查看因子分层效果)
        """
        self.backtest_mode = backtest_mode
        self.backtest_data = backtest_data
        self.fac_data = fac_data
        self.clean_fac_data() # 清理因子文件
        self.benchmark = benchmark
        self.process_mode() # 设置回测参数
        self.num_groups = num_groups
        self.group_names = list(map(lambda x:'group_'+str(x), 
                                    np.arange(self.num_groups)+1)) # 分组名
        self.tradedays = sorted(self.backtest_data.keys()) # 排序后的交易日列表
        self.sorted_keys = sorted(fac_data.keys()) # 排序后的因子文件日期
        self.month_first = set(self.get_month_first(self.tradedays)) # 每月第一天
        self.graph_path = os.path.join('output/result_graph/', self.backtest_mode) 
        self.check_path() # 检查图片储存的相对路径是否存在
            
    @staticmethod
    def fill_nan_mean(x):
        """截面填充"""
        x = np.array(x).astype(float)
        x[np.isinf(x)]= np.NAN
        x[np.isnan(x)]= np.nanmean(x)
        return x
    
    @staticmethod
    def fill_nan_zero(x):
        """暴力0填充"""
        x = np.array(x).astype(float)
        x[np.isinf(x)]= np.NAN
        x[np.isnan(x)]= 0
        return x
    
    @staticmethod
    def get_month_first(tradedays_list):
        """
        传入排序好的交易日列表 返回每月的第一天(list) O(n)
        """
        ans = []
        pre = None
        for date in tradedays_list:
            year,month,day = date.split('-')
            if pre==None:
                pass
            elif month != pre[1]:
                ans.append(date)
            pre= (year,month,day)
        return ans
    
    def check_path(self):
        """检查图片保存路径是否存在"""
        if not os.path.exists(self.graph_path):
            os.makedirs(os.path.join('./', self.graph_path))
        return 
    
    def clean_fac_data(self):
        """清理因子文件"""
        for key in self.fac_data.keys():
            # 暴力截面/0 填充
            self.fac_data[key] = self.fac_data[key].apply(self.fill_nan_mean, axis=0)
            self.fac_data[key] = self.fac_data[key].apply(self.fill_nan_zero, axis=0)
        print('因子数据清洗完毕')
    
    def process_mode(self):
        """根据交易模式设置回测参数"""
        self.interval = int(re.findall('\d+', self.backtest_mode)[-1])-1
        if self.backtest_mode[0] == 'O':
            self.price = 'open'
        else: 
            self.price = 'close'
        print('回测基本参数设置完毕')
        
    def get_price(self, stock_list, date):
        """获取当日价格"""
        data = self.backtest_data[date] #通过date 进入当日的df
        return data.loc[stock_list, self.price] # 根据设定参数返回对应column

    def get_tradeday(self, date, count):
        """获取指定间隔的交易日 向后搜索即count为正 向前搜索即count为负"""
        pos = self.tradedays.index(date)
        if pos+count >= len(self.tradedays):
            return None
        return self.tradedays[pos + count]
    
    def arr2df(self, arr):
        """
        将ret_arr 转化为df格式
        arr的形式是一个列表 每个位置上是一个字典 对应值为list
        """
        # 外循环遍历所有group 内层循环遍历所有分组 合并后取avg处理
        merge = np.array([np.average(np.concatenate([np.array(arr[i][j]) 
                          for i in range(self.interval)]).reshape(-1,self.interval),
                          axis=1) for j in range(self.num_groups)])
        df = pd.DataFrame(merge).T
        df.columns = self.group_names
        df.index = pd.to_datetime(self.sorted_keys[1:])
        return df 
    
    def list2df(self, l):
        """
        将IC_list(一个列表) 转化为df格式
        """
        df = pd.DataFrame(l)
        df.columns = ['IC']
        df.index = pd.to_datetime(self.sorted_keys[:len(l)])
        return df
    
    def make_ls_df(self, ret_df):
        """获取ret_df 并转化为多空对冲df"""
        first = ret_df.loc[:,self.group_names[0]]
        last = ret_df.loc[:,self.group_names[-1]]
        if last.cumprod().iloc[-1] > first.cumprod().iloc[-1]: # 确定多空方向
            long, short = last.values, first.values
        else: 
            long, short = first.values, last.values
        LS = long - short + 1 # 多空对冲每日收益
        LS_df = pd.DataFrame([long, short, LS], index = ['long','short','LS'], 
                             columns = ret_df.index).T
        return LS_df
        
    def loop(self, fac_to_test):
        """
        遍历整个因子文件的日期 进行回测 
        返回num_groups组收益以及benchmark收益的df 以及IC的df(1列)
        """
        # 初始化持仓 收益 IC数组 benchmark的收益率列表
        hold_arr = [collections.defaultdict(set) for _ in range(self.interval)]
        ret_arr = [collections.defaultdict(list) for _ in range(self.interval)]
        IC_list = []
        ret_benchmark = []
        
        sig = 0 # 用于确定是哪组调仓
        for date in self.sorted_keys[1:]: # 因子文件的第一天无法用于交易
            if date in self.month_first: print(date) # 月度标识
            pre = self.get_tradeday(date,-1) # 定位到上一个交易日
            # benchmark收益率 获得值本身
            ret_benchmark.append((self.get_price([self.benchmark],date)/
                                  self.get_price([self.benchmark],pre)).values[0])
            tod_chg = sig % self.interval # 当前需要调仓的组
            sig += 1  # 移动
            for idx1, hold_dic in enumerate(hold_arr): # 遍历整个持股组
                for idx2 in range(self.num_groups): # 计算当日5组收益
                    code_target = list(hold_dic[idx2]) # 当前组的持仓
                    if len(code_target)==0: # 该组尚未启动
                        ret_arr[idx1][idx2].append(1) 
                    else:
                        try:
                            ret = np.nanmean(self.get_price(code_target,date)/
                                              self.get_price(code_target, pre))
                        except:
                            ret = 1
                        if np.isnan(ret): ret = 1
                        ret_arr[idx1][idx2].append(ret)
                if idx1 == tod_chg: # 换仓组
                    hold_dic = hold_arr[idx1]
                    # 获取昨日因子数据 从小到大排列
                    single_fac = self.fac_data[pre].loc[:,fac_to_test].sort_values() 
                    code_target = list(single_fac.index)
                    group_size = int(single_fac.shape[0] / self.num_groups)
                    for idx2 in range(self.num_groups): # 更新持仓
                        pos = int(idx2*group_size)
                        hold_dic[idx2] = set(code_target[pos : pos+group_size])
                    hold_arr[idx1] = hold_dic # 存入hold_arr
                    # 计算IC
                    after = self.get_tradeday(date,self.interval) # 定位到下一个换仓日
                    if not after: pass # 无法获取价格数据
                    else:
                        try:
                            ret = (self.get_price(code_target, after)/
                                    self.get_price(code_target, date)).values
                        except:
                            ret = np.ones(shape=len(code_target))
                        IC_val = scst.spearmanr(single_fac.values, 
                                                ret,
                                                nan_policy='omit')[0]
                        IC_list.append(abs(IC_val))
        
        ret_df = self.arr2df(ret_arr) # 记录每日收益的df
        ret_df['benchmark'] =  ret_benchmark # 最后一列拼接benchmark
        IC_df = self.list2df(IC_list) # 记录每日IC的df
        
        return ret_df, IC_df
    
    def save_fig(self, ret_df, IC_df, LS_df, fac_to_test):
        """
        接收两个df (ret/IC) 
        保存五组对冲收益图 IC时序变化图 以及多空对冲图
        """
        ret_df.iloc[:,:-1].cumprod().plot(figsize=(26,15)) # 最后一列是benchmark
        plt.savefig(os.path.join(self.graph_path,f"{fac_to_test}_groups.png"))
        plt.close()
        IC_df.plot(figsize=(26,15))
        plt.savefig(os.path.join(self.graph_path, f"{fac_to_test}_IC.png"))
        plt.close()
        LS_df['LS'].cumprod().plot(figsize=(26,15))
        plt.savefig(os.path.join(self.graph_path, f"{fac_to_test}_LS.png"))
        plt.close()
        
    def stat_ret(self, ret_df, LS_df):
        """计算分组收益率指标 返回一个字典"""
        dic = {} # 初始化指标字典
        hedge_df = pd.DataFrame([LS_df['long'].values, LS_df['short'].values,
                                 ret_df['benchmark'].values], 
                                index = ['long','short','benchmark'],
                                columns = LS_df.index).T # 对冲后的df
        hedge_df['long'] = hedge_df['long'] - hedge_df['benchmark'] + 1
        hedge_df['short'] = hedge_df['short'] - hedge_df['benchmark'] + 1
        
        dic['long_return'] = LS_df.cumprod()['long'].iloc[-1] -1 
        dic['short_return'] = LS_df.cumprod()['short'].iloc[-1] -1
        dic['long_hedge_return'] =  hedge_df.cumprod()['long'].iloc[-1] -1
        dic['short_hedge_return'] = hedge_df.cumprod()['short'].iloc[-1] -1
        dic['LS_return'] = LS_df.cumprod()['LS'].iloc[-1]
        
        mul = np.sqrt(252/self.interval) # 乘数
        dic['long_sharpe'] = mul * ((np.nanmean(LS_df['long'])-1) / 
                                    (np.nanstd(LS_df['long'])+1e-9))
        dic['long_hedge_sharpe'] = mul * ((np.nanmean(hedge_df['long'])-1) / 
                                          (np.nanstd(hedge_df['long'])+1e-9))
        dic['LS_sharpe'] = mul * ((np.nanmean(LS_df['LS'])-1) / 
                                  (np.nanstd(LS_df['LS'])+1e-9))
        return dic
        
    def stat_IC(self, IC_df):
        """计算IC相关指标 返回一个字典"""
        dic = {} # 初始化指标字典
        dic['IC_mean'] = np.nanmean(IC_df['IC'])
        dic['IC_std'] = np.nanstd(IC_df['IC'])
        dic['IR'] = dic['IC_mean'] / (dic['IC_std']+1e-9)
        dic['IC_t_test'], dic['IC_p_val'] = scst.ttest_1samp(IC_df['IC'], 0)
        dic['IC_skew'] = scst.skew(IC_df['IC'])
        dic['IC_kurtosis'] = scst.kurtosis(IC_df['IC'])
        return dic
        
    def backtest(self, fac_to_test):
        """回测主函数"""
        ret_df, IC_df = self.loop(fac_to_test)
        LS_df = self.make_ls_df(ret_df)
        self.save_fig(ret_df, IC_df, LS_df, fac_to_test) # 绘图并保存
        ret_stat_dic = self.stat_ret(ret_df, LS_df) # 计算分组收益率指标
        IC_stat_dic = self.stat_IC(IC_df) # 计算IC相关指标
        stat_dic = {**ret_stat_dic, **IC_stat_dic}
                            
        return pd.DataFrame(stat_dic.values(), 
                            index = stat_dic.keys(),
                            columns = [fac_to_test]).T
        
    
if __name__ == '__main__':
    backtest_mode = 'OT1-OT11'
    benchmark = '000906.XSHG'
    num_groups = 5
    
    with open('D://alpha//stock_data//data_all_0225.pkl', 'rb') as f:
        backtest_data = pickle.load(f)
    print('股票价格数据，读取完成')
    
    with open('D:/alpha//量化投资//Z_Score_2016_2020.pkl', 'rb') as f:
        fac_data = pickle.load(f)
    print('所有因子导入')
                    
    share_fac = Manager().dict()
    for key in fac_data.keys():
        share_fac[key] = fac_data[key]
    print('共享内存成功')
    
    fac_list = fac_data[list(fac_data.keys())[0]].columns.tolist()
    print(f'测试因子个数{len(fac_list)}')
    
    # 初始化对象
    obj = BacktestEngine(backtest_mode, backtest_data, share_fac, 
                         benchmark, num_groups)
   
    # for fac in fac_list:
    #     obj.backtest(fac)
    #     break
    
    pool = Pool(processes=8)
    print("运行进程池")
    ans = pool.map(obj.backtest, fac_list)
    
    # 保存
    print("保存文件")
    file_path = f'output/{backtest_mode}.csv'
    if os.path.exists(file_path): os.remove(file_path)
    pd.concat(ans).to_csv(file_path)
    print("保存成功")