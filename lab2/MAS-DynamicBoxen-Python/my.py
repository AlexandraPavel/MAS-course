from environment import *
import time

class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, desired_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)

        self.desired_state = desired_state
        print("desired_state", desired_state)
        self.intentions = []
        self.belief = None
        self.idx = 0
        self.idx_block = 0
        self.picked_up_block = None
        self.stacks = desired_state.get_stacks()
        print("Stacks", self.stacks[0])


    def response(self, perception: BlocksWorldPerception):
        # TODO: revise beliefs; if necessary, make a plan; return an action.
        # raise NotImplementedError("not implemented yet; todo by student")
        # step 1. revise beliefs based on perceptions detecting if the blocks have been moved;
        # step 2. if the agent's beliefs do not correspond on to the agents desires
        # (constructing a stack in the final configuration)
        # make a subplan (calling the plan() method for a new desire)
        # otherwise proceed with the current plan

        self.belief = None
        self.intentions = []
        self.belief = perception.current_world
      
        # revise beliefs
        if self.belief == self.desired_state or self.idx >= len(self.stacks):
            return AgentCompleted()
        
        # make a plan
        action = self.plan()[0]
        print("My action: {}".format(action))

        return action


    def revise_beliefs(self, perceived_world_state: BlocksWorld):
        # TODO: check if what the agent knows corresponds to what the agent sees
        #raise NotImplementedError("not implemented yet; todo by student")
        pass


    def plan(self) -> List[BlocksWorldAction]:
        # TODO: return a new plan, as a sequence of `BlocksWorldAction' instances, based on the agent's knowledge.
        # 1. Put all misplaced blocks on table
        # 2. Then build goal.
        if self.idx >= len(self.stacks):
            return [AgentCompleted()]
        
        belief = self.belief.clone()

        print("stack", self.stacks[self.idx])
        print("Current belief", self.belief.stacks, belief.stacks)

        if self.idx_block >= len(self.stacks[self.idx].blocks):
            self.idx += 1
            self.idx_block = 0

        block = self.stacks[self.idx].blocks[(self.idx_block - 1)]
        print("**** Block *** ", block)

        if self.picked_up_block:
            # A != B
            if self.picked_up_block != block:
                action = [PutDown(self.picked_up_block)]
                self.picked_up_block = None
                return action
            
            if self.desired_state.get_stack(self.picked_up_block).is_on_table(self.picked_up_block):
                action = [PutDown(self.picked_up_block)]
                self.picked_up_block = None

                return action
            
            derised_stack = self.desired_state.get_stack(block)

            below_possible_block = derised_stack.get_below(block)
      
            if belief.get_stack(below_possible_block).get_top_block() == below_possible_block:
               
                action = [Stack(self.picked_up_block,  below_possible_block)]
                self.picked_up_block = None
                return action
            else:
                action = [PutDown(self.picked_up_block)]
                self.picked_up_block = None
                return action

        solving_stack = belief.get_stack(block)

        derised_stack = self.desired_state.get_stack(block)

        if solving_stack.is_locked(block):
            self.idx_block += 1
            if self.idx_block >= len(solving_stack.blocks):
                self.idx += 1
                self.idx_block = 0
            
            return [NoAction()]
        
        # verify if block on top
        if solving_stack.is_clear(block):
            below_desired_stack = self.stacks[self.idx].get_below(block)
            below_belief_stack = solving_stack.get_below(block)

            if solving_stack.is_locked(block):
                self.idx_block += 1
                if self.idx_block >= len(solving_stack.blocks):
                    self.idx += 1
                    self.idx_block = 0
                
                return [NoAction()]


            if below_belief_stack == below_desired_stack:
                crt_block = block
                flag_locked = False
                
                while not flag_locked and crt_block != solving_stack.blocks[0]:
                    prev_block = crt_block
                    crt_block = solving_stack.get_below(crt_block)
         
                    if solving_stack.is_locked(crt_block):
                        flag_locked = True
                        crt_block = prev_block
                    else:
                        self.intentions.append(Lock(crt_block))
                if crt_block == block:
                    self.idx_block += 1

                    if self.idx_block >= len(solving_stack.blocks):
                        self.idx += 1
                        self.idx_block = 0

                return [Lock(crt_block)]
            else:
                if solving_stack.is_clear(block) and solving_stack.get_below(block) == None:
                    self.picked_up_block = block
                    return [PickUp(block)]
                else:
                    self.picked_up_block = block
                    return [Unstack(block, solving_stack.blocks[-2])]
        else:
            crt_block = solving_stack.blocks[-1]
            self.picked_up_block = crt_block
            action = [Unstack(solving_stack.blocks[-1], solving_stack.blocks[-2])]

            while block != crt_block:
                prev_crt_block = crt_block
                crt_block = solving_stack.get_below(crt_block)
                self.intentions.append(Unstack(prev_crt_block, crt_block))
            
            return action

    def status_string(self):
        # TODO: return information about the agent's current state and current plan.
        str_intent = ""
        for i in range(len(self.intentions)):
            str_intent += str(self.intentions[i])
        return "Agent " + str(self) + "\nState \n" + str(self.belief) + "\nPlan " + str_intent + "\n"


class Tester(object):
    STEP_DELAY = 0.0
    TEST_SUITE = "tests/0e-large/"

    EXT = ".txt"
    SI  = "si"
    SF  = "sf"

    DYNAMICITY = .0 # set it to 0 to see if the planning is correctly, than put in to 1 if you want hte enviroment to change every step
    # the goal is to get to the solution 

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)



    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(BlocksWorld(input_stream=input_stream))


    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            print("Agent %s desires:" % str(agent))
            print(str(desires))


    def make_steps(self):
        print("\n\n================================================= INITIAL STATE:")
        print(str(self._environment))
        print("\n\n=================================================")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            print(str(self._environment))
            

            for ag in self._agents:
                print(ag.status_string())

            nr_steps += 1
            # break

            print("\n\n================================================= STEP %i completed." % nr_steps)

        print("\n\n================================================= ALL STEPS COMPLETED")





if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()