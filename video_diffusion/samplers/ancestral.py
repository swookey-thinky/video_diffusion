"""Ancestral sampling, from DDPM."""

import torch
from typing import Dict, List, Optional, Tuple

from image_diffusion.diffusion import PredictionType, DiffusionModel
from image_diffusion.samplers.base import ReverseProcessSampler
from image_diffusion.utils import dynamic_thresholding, broadcast_from_left


class AncestralSampler(ReverseProcessSampler):
    def __init__(
        self, reconstruction_guidance: bool = False, omega: float = 0.0, **kwargs
    ):
        super().__init__()

        self._reconstruction_guidance = reconstruction_guidance
        self._reconstruction_omega = omega

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        context: Dict,
        unconditional_context: Optional[Dict],
        diffusion_model: DiffusionModel,
        guidance_fn=None,
        classifier_free_guidance: Optional[float] = None,
    ):
        """Reverse process single step.

        Samples x_{t-1} given x_t - the joint distribution p_theta(x_{t-1}|x_t).

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of the current timestep.
            low_res_context: Low resolution context, for cascaded models.
            y: Tensor batch of class labels if they exist
            guidance_fn: Optional guidance function using the gradients of a classifier
                to guide diffusion.
            classifier_free_guidance: Classifier free guidance value

        Returns:
            Tensor batch of the distribution at timestep t-1.
        """
        model_mean, model_variance, model_log_variance, pred_xstart = (
            self.p_mean_variance(
                x=x,
                context=context,
                unconditional_context=unconditional_context,
                diffusion_model=diffusion_model,
                clip_denoised=True,
                classifier_free_guidance=classifier_free_guidance,
            )
        )

        # No noise if t = 0
        t = context["timestep"]
        noise = torch.randn_like(x)

        if guidance_fn is not None:
            model_mean = self._guidance_mean(
                guidance_fn, model_mean, model_variance, x, t, context["classes"]
            )

        timestep_idx = context["timestep_idx"]
        pred_img = torch.where(
            torch.tensor(timestep_idx == 0),
            pred_xstart,
            model_mean + torch.exp(0.5 * model_log_variance) * noise,
        )

        if "reconstruction_update" in context:
            x_hat_a_mean, _, logvar_a, x_hat_a = context["reconstruction_update"]
            del context["reconstruction_update"]

            context["z_a_t"] = torch.where(
                torch.tensor(timestep_idx == 0),
                x_hat_a,
                x_hat_a_mean + torch.exp(0.5 * logvar_a) * noise,
            )
        return pred_img

    def p_mean_variance(
        self,
        x,
        context: Dict,
        unconditional_context: Optional[Dict],
        diffusion_model: DiffusionModel,
        clip_denoised=True,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
        classifier_free_guidance: Optional[float] = None,
    ):
        """Calculates the mean and the variance of the reverse process distribution.

        Applies the model to get $p(x_{t-1} | x_t)$, an estimate of the reverse
        diffusion process.

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of timesteps.
            low_res_context: Low resolution context, for cascaded models.
            clip_denoised: if True, clip the denoised signal into [-1, 1].
            epsilon_v_param: If not None, a tensor batch of the model output at t.
                Used to freeze the epsilon path to prevent gradients flowing back during
                VLB loss calculation.
            y: Tensor batch of class labels if they exist
            classifier_free_guidance: Classifier free guidance value

        Returns:
            A tuple of the following values:
                mean: Tensor batch of the reverse distribution mean.
                variance: Tensor batch of the reverse distribution variance.
                log_variance: Tensor batch of the log of the reverse distribution variance.
        """
        B, C = x.shape[:2]

        assert context["timestep"].shape == (B,)

        x_hat, variance, log_variance = self._pred_x_hat(
            z_t=x,
            context=context,
            unconditional_context=unconditional_context,
            diffusion_model=diffusion_model,
            epsilon_v_param=epsilon_v_param,
            classifier_free_guidance=classifier_free_guidance,
            clip_denoised=clip_denoised,
        )

        # Add reconstruction guidance if requested. Reconstruction guidance
        # comes from Section 3.1 of "Video Diffusion Models" (https://arxiv.org/abs/2204.03458).
        # We follow Eq. (7) here, and update x_tilde_theta^b (which corresponds to
        # pred_x_start) with the second term in Eq. (7).
        if self._reconstruction_guidance and "x_a" in context:
            z_b_t = x
            x_a = context["x_a"]

            if "z_a_t" not in context:
                # Haven't noised the initial sample yet
                context["z_a_t"] = diffusion_model.noise_scheduler().q_sample(
                    x_start=x_a,
                    t=torch.ones(size=(x_a.shape[0],)),
                )
            else:
                z_a_t = context["z_a_t"]

            logsnr_t = broadcast_from_left(context["logsnr_t"], z_a_t.shape)
            alpha_t = torch.sqrt(torch.nn.functional.sigmoid(logsnr_t))

            # Calculate x_hat_a
            with torch.enable_grad():
                x_a = x_a.detach().requires_grad_(True)
                z_a_t = z_a_t.detach().requires_grad_(True)
                z_b_t = z_b_t.detach().requires_grad_(True)

                x_hat_a, variance_a, log_variance_a = self._pred_x_hat(
                    z_t=z_a_t,
                    context=context,
                    unconditional_context=unconditional_context,
                    diffusion_model=diffusion_model,
                    epsilon_v_param=epsilon_v_param,
                    classifier_free_guidance=classifier_free_guidance,
                    clip_denoised=clip_denoised,
                )
                guidance = torch.square(x_a - x_hat_a)
                guidance_factor = (
                    self._reconstruction_omega * alpha_t * 0.5
                ) * torch.autograd.grad(guidance, z_b_t)

            # Update x_hat with the guidance factor
            x_hat = x_hat - guidance_factor.detach()

            # Set parameters to update the reconstruction in the next step
            assert "reconstruction_update" not in context
            x_hat_a_mean, _, _ = diffusion_model.noise_scheduler().q_posterior(
                x_start=x_hat_a.detach(), x_t=z_a_t.detach(), context=context
            )
            context["reconstruction_update"] = (
                x_hat_a_mean,
                variance_a,
                log_variance_a,
                x_hat_a.detach(),
            )

        # Set the mean of the reverse process equal to the mean of the forward process
        # posterior.
        model_mean, _, _ = diffusion_model.noise_scheduler().q_posterior(
            x_start=x_hat, x_t=x, context=context
        )
        return model_mean, variance, log_variance, x_hat

    def _pred_x_hat(
        self,
        z_t,
        context: Dict,
        unconditional_context: Optional[Dict],
        diffusion_model: DiffusionModel,
        clip_denoised=True,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
        classifier_free_guidance: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        epsilon_theta, variance, log_variance = self._pred_epsilon(
            x=z_t,
            context=context,
            diffusion_model=diffusion_model,
            epsilon_v_param=epsilon_v_param,
        )

        # If we are using classifier free guidance, then calculate the unconditional
        # epsilon as well.
        cfg = (
            classifier_free_guidance
            if classifier_free_guidance is not None
            else diffusion_model.classifier_free_guidance()
        )

        if cfg >= 0.0 and unconditional_context is not None:
            # Unconditionally sample the model
            uncond_epsilon_theta, uncond_variance, uncond_log_variance = (
                self._pred_epsilon(
                    x=z_t,
                    context=unconditional_context,
                    epsilon_v_param=epsilon_v_param,
                )
            )
            w = cfg
            epsilon_theta = uncond_epsilon_theta + w * (
                epsilon_theta - uncond_epsilon_theta
            )

            # It's not clear from the paper how to guide learned sigma models,
            # but we will treat them the same as the epsilon-param models
            variance = uncond_variance + w * (variance - uncond_variance)
            log_variance = uncond_log_variance + w * (
                log_variance - uncond_log_variance
            )

        _maybe_clip = lambda x_: (torch.clamp(x_, -1.0, 1.0) if clip_denoised else x_)

        # Epsilon prediction (mean reparameterization)
        if diffusion_model.prediction_type() == PredictionType.EPSILON:
            pred_xstart = diffusion_model.noise_scheduler().predict_x_from_epsilon(
                z=z_t, context=context, epsilon=epsilon_theta
            )
        elif diffusion_model.prediction_type() == PredictionType.V:
            # V-prediction, from https://arxiv.org/abs/2202.00512
            pred_xstart = diffusion_model.noise_scheduler().predict_x_from_v(
                z=z_t, context=context, v=epsilon_theta
            )
        else:
            raise NotImplemented(
                f"Prediction type {diffusion_model.prediction_type()} is not implemented."
            )

        if (
            "dynamic_thresholding" in diffusion_model.config().diffusion
            and diffusion_model.config().diffusion.dynamic_thresholding.enable
        ):
            if clip_denoised:
                pred_xstart = dynamic_thresholding(
                    pred_xstart,
                    p=diffusion_model.config().diffusion.dynamic_thresholding.p,
                    c=diffusion_model.config().diffusion.dynamic_thresholding.c,
                )
        else:
            pred_xstart = _maybe_clip(pred_xstart)

        return pred_xstart, variance, log_variance

    def _pred_epsilon(
        self,
        x,
        context: Dict,
        diffusion_model: DiffusionModel,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
    ):
        """Predict the parameterized noise from the inputs.

        Args:
            x: The input images (or other data) at time t
            t: The current timestep to predict
            low_res_context: Low resolution context, for cascaded models.
            epsilon_v_param: If not None, a tensor batch of the model output at t.
                Used to freeze the epsilon path to prevent gradients flowing back during
                VLB loss calculation.
            y: Tensor batch of class labels if they exist
            image_embeddings: Tensor batch of image embeddings
            text_embeddngs: Tensor batch of text_embeddings

        Returns:
            Tuple of:
                epsilon prediction, variance, log variance
        """
        x = diffusion_model.process_input(x=x, context=context)

        model_output = (
            diffusion_model.predict_score(x, context=context)
            if epsilon_v_param is None
            else epsilon_v_param
        )

        if diffusion_model.is_learned_sigma():
            # The v_param model calculates both the reparameterized
            # mean and variance of the distribution.
            epsilon_theta, v_param = model_output

            # Calculate the learned variance from the model output.
            # We parameterize v_param as the log of the variance.
            log_variance = v_param
            variance = torch.exp(log_variance)
        else:
            epsilon_theta = model_output

            # The predicted variance is fixed. For an epsilon
            # only model, we use the "fixedlarge" estimate of
            # the variance.
            variance, log_variance = (
                diffusion_model.noise_scheduler().variance_fixed_large(context, x.shape)
            )

        return epsilon_theta, variance, log_variance

    def _guidance_mean(self, guidance_fn, p_mean, p_var, x, t, y):
        """Classifier guidance for the mean estimate.

        Compute the mean for the previous step, given a function guidance_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, guidance_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).

        Args:
            guidance_fn: Computes the gradient of a conditional log probability with
                respect to x
            p_mean: The predicted mean of the distribution
            p_var: The predicted variance of the distribution
            x: Distribution at x
            t: Timestep
            y: Class labels

        Returns:
            The mean of the distribution guided by the gradient.
        """
        gradient = guidance_fn(x, t, y)
        new_mean = p_mean.float() + p_var * gradient.float()
        return new_mean
