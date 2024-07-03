from typing import List, Dict, Any

from scipy.stats._multivariate import special_ortho_group_frozen

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage
from functools import reduce
import random

class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)
        self.offers_per_company = {}
        self.items_prices = {}

        self.contracts = {}
        random.seed(224)

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        proposed = ( 1 - (2  - auction_round) * 0.15 ) *  self.budget_dict[auction_item]
        self.items_prices[auction_item] = proposed
        return proposed
    
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
        possible_next_offer = ( 1 - pow((2 - negotiation_round) * 0.1, 3 - negotiation_round)) *  self.items_prices[negotiation_item]
        # Procents 0.6, 0.9, 1
        # print("Round", negotiation_round, "price  ", possible_next_offer, "Procent", pow((2 - negotiation_round) * 0.1, 3 - negotiation_round))

        if negotiation_round == 2:
            companies = self.offers_per_company[negotiation_item]
            sorted_offers = sorted(companies.items(), key=lambda item: item[1])
            # print("sorted_offers", sorted_offers)

            companies = []
            for x in sorted_offers:
                if x[1] == sorted_offers[0][1]:
                    companies.append(x[0])

            if negotiation_item not in self.contracts:
                self.contracts[negotiation_item] = random.choice(companies)

            print("Winner", self.contracts[negotiation_item], companies)

            if  self.contracts[negotiation_item]  == partner_agent:
                return self.offers_per_company[negotiation_item][partner_agent]
            else:
                return self.offers_per_company[negotiation_item][partner_agent] * negotiation_round
        
        if negotiation_round == 0:
            return possible_next_offer

        if self.offers_per_company[negotiation_item][partner_agent] > possible_next_offer:
            self.offers_per_company[negotiation_item][partner_agent] = possible_next_offer

        if self.items_prices[negotiation_item] < self.offers_per_company[negotiation_item][partner_agent]:
            self.offers_per_company[negotiation_item][partner_agent] = possible_next_offer

        return self.offers_per_company[negotiation_item][partner_agent]

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None: 
        self.offers_per_company[response_msg.negotiation_item][response_msg.sender] = response_msg.offer

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        self.contracts[negotiation_item] = winning_agent
        print("**** SIGNED CONTRACT for ", negotiation_item, " with ", winning_agent, " for ", winning_offer)

class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        self.acme_action = []
        self.contracts = {}
        self.participants_per_items = {}

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        if auction_item not in self.specialties:
            return False
        
        if (len(self.contracts) != 0 and self.specialties[auction_item] * 0.99 < item_budget) \
            or self.specialties[auction_item] < item_budget:
            print(self.name, "with buget",  self.specialties[auction_item], "entered for ", item_budget)
            return True
        else:
            return False

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        self.acme_action.append(auction_item)

        if num_selected == 1:
            print("Company", self.name, " won auction round ", auction_round, " for item ", auction_item ,\
                   " and is the single participant")
        else:
            self.participants_per_items[auction_item] = num_selected
            print("Company", self.name, " won auction ", auction_round, " for item ", auction_item ,\
                   " and is competing with another ", num_selected - 1, "companies")

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        print("Agent", self.name, "got offer", initiator_msg.offer, "in round", initiator_msg.round)
        signed_contracts = 0
        prices = list(self.contracts.values())
        print(prices)

        for i in prices:
            if i != 'LOST':
                signed_contracts += i

        keys = [key for key, value in self.contracts.items() if value == 'LOST']

        if self.specialties[initiator_msg.negotiation_item] < initiator_msg.offer:
            if initiator_msg.negotiation_item in self.participants_per_items:
                contra_offer = initiator_msg.offer *  0.96 * pow(0.95 - self.participants_per_items[initiator_msg.negotiation_item] * 0.01, initiator_msg.round)
                if contra_offer - signed_contracts >=  self.specialties[initiator_msg.negotiation_item]:
                    return contra_offer

            else:
                contra_offer = initiator_msg.offer *  pow(0.99, initiator_msg.round)

            if contra_offer + signed_contracts >=  self.specialties[initiator_msg.negotiation_item]:
                return contra_offer
            else:
                return self.specialties[initiator_msg.negotiation_item] * pow(1 + (2 - initiator_msg.round) * 0.01,  (2 - initiator_msg.round))
        
        else:
            if 'LOST' in self.contracts.values() or len(self.contracts) == 0:
                contra_offer = initiator_msg.offer *  0.96 * pow(0.95 - len(keys) * 0.01, initiator_msg.round)
                if contra_offer - signed_contracts >=  self.specialties[initiator_msg.negotiation_item]:
                    return contra_offer

            elif initiator_msg.negotiation_item in self.participants_per_items:
                contra_offer = initiator_msg.offer *  0.96 * pow(0.95 - self.participants_per_items[initiator_msg.negotiation_item] * 0.01, initiator_msg.round)
                if contra_offer - signed_contracts >=  self.specialties[initiator_msg.negotiation_item]:
                    return contra_offer

            else:
                contra_offer = initiator_msg.offer *  pow(0.99, initiator_msg.round)

            if contra_offer + signed_contracts >=  self.specialties[initiator_msg.negotiation_item]:
                return contra_offer
            else:
                return self.specialties[initiator_msg.negotiation_item] * pow(1 + (2 - initiator_msg.round) * 0.01,  (2 - initiator_msg.round))

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        print("***** CONTRACT ASSIGNED ***** to ", self.name, "for ", construction_item, " getting ", price,  "as revenue")
        self.contracts[construction_item] = price - self.specialties[construction_item]

    def notify_negotiation_lost(self, construction_item: str) -> None:
        self.contracts[construction_item] = 'LOST'
