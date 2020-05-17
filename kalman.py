# Kalman Filter class

class kalman(object):

    def __init__(self, process_var,est_meas_var,post_est,post_err_est ):
        self.process_var = process_var
        self.est_meas_var = est_meas_var
        self.post_est = post_est      # x_hat[0] = 0
        self.post_err_est = post_err_est   # P[0] = 1

    def update_meas(self,measurement,dt,velocity,update = True):

        B = -dt
        u = velocity
        if update == False:
            prior_est = self.post_est + B*u
            prior_err_est = self.post_err_est + self.process_var
            return prior_est, prior_err_est


        if update == True:
            prior_est = self.post_est + B*u
            prior_err_est = self.post_err_est + self.process_var

            post_err_est = (prior_err_est + (B*self.process_var*B) + (
                                self.est_meas_var)**-1 )**-1
            post_est = prior_est + post_err_est*(self.est_meas_var**-1)*(
                            measurement - prior_est)

            return post_est, post_err_est

    def latest_estimated_meas(self):
        return self.post_est
