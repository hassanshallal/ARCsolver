# this is a class that wraps important information in a single object to be passed to anyother class

class Communication:
    def __init__(self, traininputs, trainoutputs, testinputs, bg, objective_status, objective, assignments_leads, asssignments_output, token_to_colors, test_token_to_color, testoutputs = None):
        self.traininputs = traininputs
        self.trainoutputs = trainoutputs
        self.testinputs = testinputs
        self.bg = bg

        self.objective_status = objective_status
        self.objective = objective
        self.assignments_leads = assignments_leads
        self.asssignments_output = asssignments_output
        self.token_to_colors = token_to_colors
        self.test_token_to_color = test_token_to_color
        self.testoutputs = testoutputs
