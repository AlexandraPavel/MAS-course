from typing import List, Dict, Any

from scipy.stats._multivariate import special_ortho_group_frozen

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage
from functools import reduce


class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)
        self.offers_per_company = {}

        self.contracts = {}

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        return ( 1 - (2  - auction_round) * 0.1 ) *  self.budget_dict[auction_item]
    
    def add_companies_list(self, auction_item: str, companies: List[str]) -> None:
        if auction_item not in self.offers_per_company:
            self.offers_per_company[auction_item] = {}

        self.offers_per_company[auction_item] = dict((el, 0) for el in companies)

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        if len(responding_agents) == 0:
            if auction_item not in self.offers_per_company:
                self.offers_per_company[auction_item] = {}
            print("Item ", auction_item, " for round ", auction_round, " there are no participants")
        else:
            agents_entering_the_bid = reduce(lambda x, y: x + ", " + y, responding_agents)
            self.add_companies_list(auction_item, responding_agents)
     
            print("Item ", auction_item, " for round ", auction_round, \
                  " the bid has the following participants: ", agents_entering_the_bid)

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        possible_next_offer = ( 1 - pow((2 - negotiation_round), 2) * 0.1 ) *  self.budget_dict[negotiation_item]
        # Procents 0.6, 0.9, 1
        # print("Round", negotiation_round, "Procent ", ( 1 - pow((2 - negotiation_round), 2) * 0.1 ))

        if negotiation_round == 2:
            return self.offers_per_company[negotiation_item][partner_agent]
        
        if negotiation_round == 0:
            return possible_next_offer

        if self.offers_per_company[negotiation_item][partner_agent] > possible_next_offer:
            self.offers_per_company[negotiation_item][partner_agent] = possible_next_offer
        else:
            self.offers_per_company[negotiation_item][partner_agent] *= ( 1 - pow((2 - negotiation_round), 2) * 0.01 )

        return self.offers_per_company[negotiation_item][partner_agent]

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None: 
        self.offers_per_company[response_msg.negotiation_item][response_msg.sender] = response_msg.offer

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        self.contracts[negotiation_item] = \
            {
                'winner': winning_agent,
                'offer': winning_offer
            }
        print("**** SIGNED CONTRACT for ", negotiation_item, " with ", winning_agent, " for ", winning_offer)

class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        self.acme_action = []
        self.contracts = {}

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        if auction_item not in self.specialties:
            return False
        
        if self.specialties[auction_item] <= item_budget:
            return True
        else:
            return False

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        self.acme_action.append(auction_item)

        if num_selected == 1:
            print("Company", self.name, " won auction round ", auction_round, " for item ", auction_item ,\
                   " and is the single participant")
        else:
            print("Company", self.name, " won auction ", auction_round, " for item ", auction_item ,\
                   " and is competing with another ", num_selected - 1, "companies")

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        print("Agent", self.name, "got offer", initiator_msg.offer, "in round", initiator_msg.round)
        if initiator_msg.offer >= self.specialties[initiator_msg.negotiation_item]:
            return initiator_msg.offer
    
        keys = [key for key, value in self.contracts.items() if value == 'LOST']
            
        if 'LOST' in self.contracts.values() or len(self.contracts) == 0:
            return self.specialties[initiator_msg.negotiation_item] *  0.96 * pow(0.95 - len(keys) * 0.01, initiator_msg.round)
        return self.specialties[initiator_msg.negotiation_item] *  pow(0.99, initiator_msg.round)

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        print("***** CONTRACT ASSIGNED ***** to ", self.name, "for ", construction_item, " getting ", price,  "as revenue")
        self.contracts[construction_item] = price

    def notify_negotiation_lost(self, construction_item: str) -> None:
        self.contracts[construction_item] = 'LOST'
