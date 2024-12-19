from pydantic import BaseModel
import json
import time
import copy
import os
import pandas as pd
import numpy as np

# Design a basic LogRound class
class LogRound(BaseModel):
    round_id :int = 0 # 标注是哪一轮的  
    # 所有轮数的log
    # round_id : {log_round,log_social_network}
    log: dict = {}

    
    save_dir:str=""
    
    
    @classmethod
    def load_from_json(cls,
                       json_path):
        with open(json_path,'r',encoding = 'utf-8') as f:
            log=json.load(f)
        
        round_id = list(log.keys())[-1]
        
        if round_id !="group":
            round_id = int(round_id)
        else:
            round_id = 0
            
        return cls(log = log,
                   round_id = round_id,
                   save_dir = json_path)
    
    def step(self, round_id):
        self.round_id = round_id
        self.log[self.round_id] = {} # 下一轮log的 initialize
    
    def set_one_tenant_choose_process(self,
                                      tenant_id,
                                      log_tenant):
        if self.round_id > 0:
            if "log_round" not in self.log[self.round_id].keys():
                self.log[self.round_id]["log_round"] = {}
            if "log_round_prompts" not in self.log[self.round_id].keys():
                self.log[self.round_id]["log_round_prompts"] = {}
            if tenant_id in self.log[self.round_id]["log_round"].keys():
                self.log[self.round_id]["log_round"][tenant_id].update(copy.deepcopy(log_tenant.log_round))
            else:
                self.log[self.round_id]["log_round"][tenant_id] = copy.deepcopy(log_tenant.log_round)
            if tenant_id in self.log[self.round_id]["log_round_prompts"].keys():
                self.log[self.round_id]["log_round_prompts"][tenant_id].update(copy.deepcopy(log_tenant.log_round_prompts))
            else:
                self.log[self.round_id]["log_round_prompts"][tenant_id] = copy.deepcopy(log_tenant.log_round_prompts)
            log_tenant.reset()
    
    def set_group_log(self,
                      tenant,
                      log_tenant):
        if "group" not in self.log.keys():
            self.log["group"] = {}
        self.log["group"][tenant.id] = {
            "log_round" : copy.deepcopy(log_tenant.log_round),
            "log_round_prompts": copy.deepcopy(log_tenant.log_round_prompts),
            "queue_name":tenant.queue_name
        }
        log_tenant.reset()
    
    
            
    def set_social_network_mem(self,
                               social_network_mem:dict):
        if "log_social_network" not in self.log[self.round_id].keys():
            self.log[self.round_id]["log_social_network"] = {}
        self.log[self.round_id]["log_social_network"]["social_network_mem"] = \
            copy.deepcopy(social_network_mem)
        

    def save_data(self):
        if not os.path.exists(os.path.dirname(self.save_dir)):
            os.makedirs(os.path.dirname(self.save_dir))
        
        with open(self.save_dir, 'w', encoding='utf-8') as file:
            json.dump(self.log, file, indent=4,separators=(',', ':'),ensure_ascii=False)

    def reset(self):
        self.log={}
        
        
    def count_utility(self,
                      utility_choosed:pd.DataFrame,
                      system,
                      tenant_manager,
                      type_utility = "all"):
        
        save_dir = os.path.dirname(self.save_dir)
        save_dir = os.path.join(save_dir,type_utility)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
   
        utility_matrix = pd.DataFrame()
        for tenant_id,utility_one in utility_choosed.items():
            for k,v in utility_one.items():
                utility_matrix.loc[tenant_id,k] = v

        
        utility_eval_matrix = pd.DataFrame()

        """各个组内的公平性、满意度打分"""
        utility_grouped = utility_matrix.groupby(by = ["group_id"])
        
        for group_id, group_utility in utility_grouped:
            # 公平度
            
            scores = group_utility["choose_u"]
            
            utility_eval_matrix.loc[f"least_misery",group_id] = min(scores)
            utility_eval_matrix.loc[f"variance",group_id] = np.var(scores)
            utility_eval_matrix.loc[f"jain'sfair",group_id] = np.square(np.sum(scores))/(np.sum(np.square(scores)) * utility_matrix.shape[0])
            utility_eval_matrix.loc[f"min_max_ratio",group_id] = np.min(scores)/np.max(scores)
            
            # 满意度
            utility_eval_matrix.loc[f"sw",group_id] = np.sum(scores)
                
        """整体的公平性和满意度"""
        scores = utility_matrix["choose_u"]
        utility_eval_matrix.loc[f"least_misery","all"] = min(scores)
        utility_eval_matrix.loc[f"variance","all"] = np.var(scores)
         
        # 满意度

        utility_eval_matrix.loc[f"sw","all"] = np.sum(scores)

                
        """弱势群体的公平度"""
        # utility_grouped = utility_matrix.groupby(by = ["priority"])
        # for priority_lable, group_utility in utility_grouped:
        #     scores = group_utility["choose_u"]
        utility_p = utility_matrix[utility_matrix["priority"]]
        utility_np = utility_matrix[utility_matrix["priority"]!= True]
        if utility_p.shape[0]!= 0 and utility_np.shape[0]!=0:
            utility_eval_matrix.loc["F(W,G)","all"] = np.sum(utility_p["choose_u"])/utility_p.shape[0] -\
            np.sum(utility_np["choose_u"])/utility_np.shape[0]
        
        
        """计算基尼指数,原本的定义是将收入分配作为输入,
        这里为了衡量公平性, 将房屋分配的utitlity作为输入"""
        # Calculate Gini coefficient and Lorenz curve coordinates
        gini, x, y = self.calculate_gini(utility_matrix["choose_u"])
        import matplotlib.pyplot as plt

        # Plot the Lorenz curve
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        plt.plot([0, 1], [0, 1], linestyle='--', color='k')
        plt.fill_between(x, x, y, color='lightgray')
        plt.xlabel("Cumulative % of Population")
        plt.ylabel("Cumulative % of Income/Wealth")
        plt.title(f"Lorenz Curve (Gini Index: {gini:.2f})")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir,"GINI_index.pdf"))
        
        utility_eval_matrix.loc["GINI_index","all"] = gini

        
        
        """一些客观的指标（例如人均住宅面积）"""
        objective_evaluation = pd.DataFrame()
        for tenant_id, choosed_info in utility_choosed.items():
            house_id  = choosed_info["choose_house_id"]
            utility_matrix.loc[tenant_id,"family_members_num"] = tenant_manager.total_tenant_datas[tenant_id].family_num
            try:
                house_size = system.house_manager.data.get(house_id).get("house_area")
                house_size = float(house_size.strip())
                utility_matrix.loc[tenant_id,"house_size"] = house_size

                
            except Exception as e:
                utility_matrix.loc[tenant_id,"house_size"] = 0
        
        utility_matrix_objective = utility_matrix[utility_matrix["house_size"]!=None]
        utility_matrix_objective["avg_area"] = utility_matrix_objective["house_size"]/utility_matrix_objective["family_members_num"]
        
        utility_matrix_objective_grouped = utility_matrix_objective.groupby(by = ["group_id"]) 
        
        for group_id,group_matrix in utility_matrix_objective_grouped:
            objective_evaluation.loc["mean_house_area",group_id] = np.average(group_matrix["avg_area"])
            objective_evaluation.loc["mean_wait_turn",group_id] = np.average(group_matrix["wait_turn"])
            objective_evaluation.loc["mean_idle_wait_turn",group_id] = np.average(group_matrix["IWT"])
            
        objective_evaluation.loc["mean_house_area","all"] = np.average(utility_matrix_objective["avg_area"])
        objective_evaluation.loc["var_mean_house_area","all"] = np.var(utility_matrix_objective["avg_area"])
        objective_evaluation.loc["mean_wait_turn","all"] = np.average(utility_matrix_objective["wait_turn"])
        objective_evaluation.loc["mean_idle_wait_turn","all"] = np.average(utility_matrix_objective["IWT"])
        
        
        # 计算逆序对
        count_rop = 0
        for tenant_id_a in utility_choosed.keys():
            for tenant_id_b in utility_choosed.keys():
                if (tenant_manager.total_tenant_datas[tenant_id_a].family_num < \
                tenant_manager.total_tenant_datas[tenant_id_b].family_num ) and \
                    (utility_matrix.loc[tenant_id_a,"house_size"]>
                     utility_matrix.loc[tenant_id_b,"house_size"]):
                    count_rop+=1
        objective_evaluation.loc["Rop","all"] = count_rop
        objective_evaluation.index.name = 'type_indicator'
        
        # 设置指标的标签
        index_map ={
            "Satisfaction":["sw"],
            "Fairness":["least_misery","variance","jain'sfair","min_max_ratio","F(W,G)","GINI_index"]
        } 
        
        index_ori = utility_eval_matrix.index
        index_transfered = [index_ori,[]]
        for index_one in index_ori:
            for k_type, type_list in index_map.items():
                if index_one in type_list:
                    index_transfered[1].append(k_type)
                    break
                
        utility_eval_matrix.index = pd.MultiIndex.from_arrays(
            index_transfered, names=('type_indicator', 'eval_type'))
        # utility_eval_matrix.index = index
        utility_eval_matrix.to_csv(os.path.join(save_dir,"utility_eval_matrix.csv"))
        objective_evaluation.to_csv(os.path.join(save_dir,"objective_evaluation_matrix.csv"))
        utility_matrix_objective.index.name = "tenant_ids"
        utility_matrix_objective.to_csv(os.path.join(save_dir,"utility_matrix_objective.csv"))
        
        
        
    def group(self,tenant)->str:
        """return the group of tenant in evaluation"""
        if tenant.family_num>3:
            return  "family_num>3"
        elif tenant.family_num >1: 
            return "3>=family_num>=2"
        else:
            return "family_num=1"
    
    
    def plt_tenant_choosing_distribution(self):
        
        
        pass # 误差图片
    
    def evaluation_matrix(self,
                          global_score,
                          system,
                          max_count_rounds = 10 ): # 评价系统的公平度，满意度
        
        tenant_manager = global_score.tenant_manager
        """this function must be called at the end of running this system."""
        
        ### filter log
        filtered_log = {}
        for log_id, log in self.log.items():
            if log_id == "group":
                filtered_log[log_id] = log
            elif int(log_id) <= max_count_rounds:
                filtered_log[log_id] = log
                
        self.log = filtered_log
        utility = global_score.get_result()
        utility_choosed = {}
        
        
        for log_id,log in self.log.items():
            if log_id == "group":
                continue
            if log == {}:
                continue 
            
            log_round = log.get("log_round")
            if log_round is None:
                continue
            for tenant_id, tenant_info in log_round.items():
                 
                if "choose_house_id" in tenant_info.keys():
                    if tenant_info["choose_house_id"] !="None":
                        rating_score_choose_u = utility[str(tenant_id)]["ratings"][tenant_info["choose_house_id"]]
                        # group_id_t = self.log["group"][tenant_id]["queue_name"]
                        tenant = tenant_manager.total_tenant_datas[tenant_id]
                        group_id_t = self.group(tenant)
                        assert tenant_id not in utility_choosed.keys(),f"Error!! Tenant {tenant_id} chosing house twice."
                        enter_turn = tenant_manager.get_tenant_enter_turn(tenant)
                        
                        utility_choosed[tenant_id] = {
                                    "choose_u": rating_score_choose_u["score"],
                                    "group_id": group_id_t,
                                    "priority": not all(not value for value in tenant.priority_item.values()),
                                    "choose_house_id": tenant_info["choose_house_id"],
                                    "wait_turn": int(log_id) - int(enter_turn)
                                } 
                
        # 加入idle等待turn
        for log_id,log in self.log.items():      
            if log_id == "group":
                continue
            if log == {}:
                continue 
            log_round = log.get("log_round")
            if log_round is None:
                continue
            
            for tenant_id, tenant_info in log_round.items():
                if tenant_id in utility_choosed.keys() and \
                    utility_choosed[tenant_id].get("IWT",None) is None: 
                        
                    tenant = tenant_manager.total_tenant_datas[tenant_id]
                    enter_turn = tenant_manager.get_tenant_enter_turn(tenant)
                    offer_turn = int(log_id)
                    utility_choosed[tenant_id]["IWT"] = int(offer_turn) - int(enter_turn)        
        
        self.count_utility(utility_choosed,system,tenant_manager,"choosed")
        
        max_turn = list(self.log.keys())[-1]
        if self.log[max_turn] == {}:
            max_turn = int(max_turn) -1
        else: max_turn = int(max_turn)
        
        tenants_system = [] # 存储所有进入了系统的tenant的id
        for turn in range(max_turn):
            if str(turn) in tenant_manager.distribution_batch_data.keys():
                tenants_system.extend(tenant_manager.distribution_batch_data[str(turn)])

        # check 
        for tenant_id in utility_choosed.keys():
            assert tenant_id in tenants_system,tenant_id
        

        for tenant_id in tenants_system:
            if tenant_id not in utility_choosed.keys():
                tenant = tenant_manager.total_tenant_datas[tenant_id]
                group_id_t = self.group(tenant)
                enter_turn = tenant_manager.get_tenant_enter_turn(tenant)
                utility_choosed[tenant_id] = {
                                    "choose_u": 0,
                                    "group_id": group_id_t,
                                    "priority": not all(not value for value in tenant.priority_item.values()),
                                    "choose_house_id": "None",
                                    "wait_turn": max_turn - int(enter_turn)
                                } 
        
        # 加入idle等待turn
        for log_id,log in self.log.items():      
            if log_id == "group":
                continue
            if log == {}:
                continue 
            log_round = log.get("log_round")
            if log_round is None:
                continue
            
            for tenant_id, tenant_info in log_round.items():
                if utility_choosed[tenant_id].get("IWT",None) is None: 
                    tenant = tenant_manager.total_tenant_datas[tenant_id]
                    enter_turn = tenant_manager.get_tenant_enter_turn(tenant)
                    offer_turn = int(log_id)
                    utility_choosed[tenant_id]["IWT"] = int(offer_turn) - int(enter_turn)
                    
        for tenant_id in utility_choosed.keys():
            if utility_choosed[tenant_id].get("IWT",None) is None: 
                utility_choosed[tenant_id]["IWT"] = utility_choosed[tenant_id]["wait_turn"]
        
        self.count_utility(utility_choosed,system,tenant_manager,"all")
        
        
    def calculate_gini(self,data):
        import numpy as np
        # Sort the data in ascending order
        data = np.sort(data)
        
        # Calculate the cumulative proportion of income/wealth
        cumulative_income = np.cumsum(data)
        
        # Calculate the Lorenz curve coordinates
        x = np.arange(1, len(data) + 1) / len(data)
        y = cumulative_income / np.sum(data)
        
        # Calculate the area under the Lorenz curve (A)
        area_under_curve = np.trapz(y, x)
        
        # Calculate the area under the line of perfect equality (B)
        area_perfect_equality = 0.5
        
        # Calculate the Gini coefficient
        gini_coefficient = (area_perfect_equality - area_under_curve) / area_perfect_equality
        
        return gini_coefficient, x, y
    
    
