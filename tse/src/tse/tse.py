# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class TrainingSpeedEstimator:
    """Implements training speed estimators from
    Ru, Robin, et al. "Speedy Performance Estimation for Neural Architecture Search."
    Advances in Neural Information Processing Systems 34 (2021).
    """

    def __init__(self, E=1, gamma=0.999, epoch_cut=10, norm=True, burn_in=2):
        """
        Parameters
        ----------
        (For classic TSE)
        E: int
            number of "burn-in" epochs to throw away at beginning of training
            for TSE-E estimator

        gamma: float
            hyperparam for exponential moving average


        (For modified TSE based on loss curve fitting)
        epoch_cutoff: int
            number of epochs worth of data used to fit loss curve;
            (e.g. if user trains for just 10 epochs instead of all the way until convergence,
            then epoch_cutoff = 10 and all step/iteration level training loss data are used within
            those 10 epochs for curve fitting)

        normalize: bool
            boolean for whether or not to normalize loss data before fitting curve; normalization
            is simply dividing by the maximum loss value and generally gives better results (default True)

        burn_in_ep: int
            number of epochs to ignore when summing calculated gradients of fitted loss curve
            (i.e. to burn in); aims to reduce noise from calculations given the variance of train loss
            during the early stages of training as well as on the iteration/step level rather than epoch

        """

        self.E = E
        self.gamma = gamma
        self.epoch_cutoff = epoch_cut
        self.normalize = norm
        self.burn_in_ep = burn_in

    def estimate(self, df_train, df_energy, T):
        """
        Parameters
        ---------
        df: Pandas dataframe
            dataframe with 'epoch' and 'train_loss' columns
        T: int
            number of epochs to consider in estimation
        Returns
        -------
        tse_dict: dict
            Results from three TSE estimation methods for training loss curve
        """

        B = len(
            df_train[df_train["epoch"] == 0]
        )  # number of steps (minibatches) in an epoch
        T_end = df_train.iloc[-1].epoch + 1  # number of total epochs

        tse = df_train[df_train["epoch"] < T].train_loss.sum() / B

        tsee = (
            df_train[
                (df_train["epoch"] >= T - self.E + 1) & (df_train["epoch"] <= T)
            ].train_loss.sum()
            / B
        )

        tseema = 0
        for t in range(0, T + 1):
            sum_losses = (
                df_train[df_train["epoch"] == t].train_loss.sum()
                / B
                * self.gamma ** (T - t)
            )
            tseema += sum_losses

        if df_energy is not None:
            energies = []
            for idx in df_energy[" index"].unique():
                df0 = df_energy[df_energy[" index"] == idx].reset_index(
                    drop=True
                )  # power by GPU device index
                E = self._compute_energy(df0)
                energies.append(E)
            total_energy = np.sum(energies) / 1e3
        else:
            total_energy = 0

        energy_per_epoch = total_energy / T_end
        energy_per_step = total_energy / len(df_train)
        tpe_dict = {
            "tse": tse,
            "tsee": tsee,
            "tseema": tseema,
            "T_end": T_end,
            "energy_per_epoch (kJ)": energy_per_epoch,
            "energy_per_step (kJ)": energy_per_step,
        }

        return tpe_dict

    def _compute_energy(self, df):
        ts = pd.to_datetime(df["timestamp"])
        ts = ts - ts[0]
        ts = ts.dt.total_seconds().to_numpy()
        # Quadrature by trapezoidal rule
        deltas = ts[1:] - ts[0:-1]
        power = df[" power.draw [W]"].to_numpy()
        avg_powers = 0.5 * (power[1:] + power[0:-1])
        energy = deltas * avg_powers  # units of watts * seconds
        return np.sum(energy)

    def estimate_acc(self, df, T):
        """
        Parameters
        ---------
        df: Pandas dataframe
            dataframe with 'epoch' and 'train_loss' columns
        T: int
            number of epochs to consider in estimation
        Returns
        -------
        tse_dict: dict
            Results from three TSE estimation methods for training accuracy curve
        """

        B = len(df[df["epoch"] == T])
        tse_acc = df[df["epoch"] < T].train_acc_stp.sum() / B

        tse_e_acc = (
            df[(df["epoch"] >= T - self.E + 1) & (df["epoch"] <= T)].train_acc_stp.sum()
            / B
        )

        tse_ema_acc = 0
        for t in range(0, T + 1):
            sum_losses = (
                df[df["epoch"] == t].train_acc_stp.sum() / B * self.gamma ** (T - t)
            )
            tse_ema_acc += sum_losses

        tse_dict_acc = {
            "TSE_acc": tse_acc,
            "TSE-E_acc": tse_e_acc,
            "TSE-EMA_acc": tse_ema_acc,
        }

        return tse_dict_acc

    def estimate_deltagrad(self, df, T):
        """
        Parameters
        ---------
        df: Pandas dataframe
            dataframe with 'epoch' and 'train_loss' columns
        T: int
            number of epochs to consider in estimation
        Returns
        -------
        tse_dict: dict
            Results from three TSE estimation methods for training accuracy curve
        """

        B = len(df[df["epoch"] == T])
        rel_df = df

        ## finite diff appoximation of discrete gradient
        rel_df["train_loss_stp_d1"] = rel_df["train_acc_stp"].diff()
        rel_df["train_loss_stp_d2"] = (
            rel_df["train_loss"]
            - 2 * rel_df["train_loss"].shift(1)
            + rel_df["train_loss_stp"].shift(2)
        )

        tse_d1 = rel_df[rel_df["epoch"] < T].train_loss_stp_d1.sum() / B
        tse_d2 = rel_df[rel_df["epoch"] < T].train_loss_stp_d2.sum() / B

        tse_ed1 = rel_df[
            (rel_df["epoch"] >= T - self.E + 1) & (rel_df["epoch"] <= T)
        ].train_loss_stp_d1.sum() / (B - 1)
        tse_ed2 = rel_df[
            (rel_df["epoch"] >= T - self.E + 1) & (rel_df["epoch"] <= T)
        ].train_loss_stp_d2.sum() / (B - 2)

        tse_emad1 = 0
        tse_emad2 = 0
        for t in range(0, T + 1):
            sum_lossesd1 = (
                rel_df[rel_df["epoch"] == t].train_loss_stp_d1.sum() / (B - 1)
            ) * self.gamma ** (T - t)
            sum_lossesd2 = (
                rel_df[rel_df["epoch"] == t].train_loss_stp_d2.sum() / (B - 2)
            ) * self.gamma ** (T - t)
            tse_emad1 += sum_lossesd1
            tse_emad2 += sum_lossesd2

        tse_dict_deltagrad = {
            "TSE_del1": tse_d1,
            "TSE-E_del1": tse_ed1,
            "TSE-EMA_del1": tse_emad1,
            "TSE_del2": tse_d2,
            "TSE-E_del2": tse_ed2,
            "TSE-EMA_del2": tse_emad2,
        }

        return tse_dict_deltagrad

    def estimate_losscurve(self, df):
        def fn(k, b1, b2, b3):
            return (1 / (b1 + b2 * k)) + b3

        def moving_average(x, w):
            return np.convolve(x, np.ones(w), "valid") / w

        def calc_gradients(x_eval, b1, b2, b3, order=1):
            grad = -1 * (b2) / ((b1 + b2 * x_eval) ** 2)
            if order == 1:
                return grad
            elif order == 2:
                return -1 * grad * ((2 * b2) / (b1 + b2 * x_eval))
            else:
                raise Exception("Available choices include 1 or 2 for order")

        def fit_nnls(
            df,
            epoch_cutoff,
            metric="trnloss",
            step_avg=False,
            roll_avg=False,
            normalize=False,
        ):
            """
            Fits non-negative least squares to data
            Inputs:
            step_avg: averages step-level loss data up to epoch level
            roll_avg: creates moving average of loss data
            normalize: normalizes loss data by respective maximum value(s)
            """

            x = df[df["epoch"] < epoch_cutoff]
            x = x.copy()

            if metric == "trnloss":
                if normalize:
                    x["train_loss"] = x["train_loss"] / x["train_loss"].max()
                if step_avg:
                    y = (
                        x.groupby("epoch")
                        .agg({"train_loss": "mean"})
                        .reset_index()["train_loss"]
                    )
                elif roll_avg:
                    #             y = moving_average(np.array(x["train_loss"]), len(x["train_loss"]))
                    y = moving_average(np.array(x["train_loss"]), 10)
                else:
                    y = np.array(x["train_loss"])

            elif metric == "trnacc":
                if step_avg:
                    y = x.groupby("epoch").agg({"train_acc_stp": "mean"}).reset_index()
                elif roll_avg:
                    #             y = moving_average(np.array(x["train_acc_stp"]), len(x["train_acc_stp"]))
                    y = moving_average(np.array(x["train_loss"]), 10)
                else:
                    y = np.array(x["train_acc_stp"])

            k = np.arange(1, len(y) + 1)

            param, _ = curve_fit(fn, k, y, bounds=(0, np.inf), maxfev=2000)
            n = len(k)

            return param, n

        ##############################

        if self.burn_in_ep >= self.epoch_cutoff:
            raise Exception(
                "Number of epochs to burn cannot be greater than epoch cutoff"
            )

        params, n = fit_nnls(
            df, epoch_cutoff=self.epoch_cutoff, normalize=self.normalize
        )

        b1, b2, b3 = params
        niters = np.arange(1, n + 1)

        max_renorm = df[df["epoch"] < self.epoch_cutoff]["train_loss"].max()
        g1 = calc_gradients(niters, b1, b2, b3, order=1)
        g2 = calc_gradients(niters, b1, b2, b3, order=2)

        fhat = fn(niters, b1, b2, b3)

        if self.normalize:
            g1 = g1 * max_renorm
            g2 = g2 * max_renorm
            fhat = max_renorm * fhat

        n_steps_ep = int(n / self.epoch_cutoff)
        steps_burn = self.burn_in_ep * n_steps_ep

        g1_avg = np.mean(g1[steps_burn:])
        g1_normavg = np.mean(g1[steps_burn:] / np.sqrt(g2[steps_burn:]))

        g1_avg_ema = 0
        g1_normavg_ema = 0

        g1burn = g1[steps_burn:]
        g2burn = g2[steps_burn:]

        T = self.epoch_cutoff
        for t in range(1, T + 1 - int(self.burn_in_ep)):

            temp1 = g1burn[((t - 1) * n_steps_ep) : (t * n_steps_ep)]
            temp2 = g2burn[((t - 1) * n_steps_ep) : (t * n_steps_ep)]

            sum_g1 = (np.sum(temp1) / n_steps_ep) * self.gamma ** (T - t)
            sum_g1norm = (np.sum(temp2) / n_steps_ep) * self.gamma ** (T - t)

            g1_avg_ema += sum_g1
            g1_normavg_ema += sum_g1norm

        grad_est_dict = {
            "d1_sum": g1_avg,
            "d1/sqrt(d2)_sum": g1_normavg,
            "d1_sum_ema": g1_avg_ema,
            "d1/sqrt(d2)_sum_ema": g1_normavg_ema,
        }

        return grad_est_dict, params, fhat, g1, g2
