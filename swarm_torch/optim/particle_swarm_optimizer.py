import torch
import torch.nn as nn


class ParticleSwarmOptimizer:
    def __init__(
        self,
        model: nn.Module,
        loss_function: object,
        num_particles: int,
        inertia_weight: float,
        cognitive_coefficient: float,
        social_coefficient: float,
        learning_rate: float,
    ) -> None:
        self._model = model
        self._loss_function = loss_function
        self._num_particles = num_particles
        self._inertia_weight = inertia_weight
        self._cognitive_coefficient = cognitive_coefficient
        self._social_coefficient = social_coefficient
        self._learning_rate = learning_rate
        self._velocity = {}
        self._current_position = {}
        self._best_position = {}
        self._global_best_position = {}
        self._best_loss = None
        self._global_best_loss = None
        self._dtype = None
        self._device = None
        self._initialize_particles()

    @property
    def best_position(self):
        return self._global_best_position

    @property
    def best_loss(self):
        return self._global_best_loss

    def _initialize_particles(self) -> None:
        state_dict = self._model.state_dict()

        for name, parameters in state_dict.items():
            self._velocity[name] = torch.stack(
                [parameters for _ in range(self._num_particles)])
            self._current_position[name] = torch.stack(
                [parameters for _ in range(self._num_particles)])

            nn.init.uniform_(self._velocity[name], -2.0, 2.0)
            nn.init.uniform_(self._current_position[name], -1.0, 1.0)

    def _initialize_best_results(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        self._best_position = self._current_position
        self._best_loss = torch.empty(
            self._num_particles, dtype=self._dtype, device=self._device)

        for i in range(self._num_particles):
            state_dict = {name: parameters[i] for name,
                          parameters in self._best_position.items()}

            self._model.load_state_dict(state_dict)

            with torch.no_grad():
                outputs = self._model(inputs)

            self._best_loss[i] = self._loss_function(outputs, target)

            if self._global_best_loss is not None:
                if self._best_loss[i] < self._global_best_loss:
                    self._global_best_position = state_dict
                    self._global_best_loss = self._best_loss[i]
            else:
                self._global_best_position = state_dict
                self._global_best_loss = self._best_loss[i]

    def _calculate_new_position(self) -> None:
        random_cognitive_coefficent = torch.empty(
            1, dtype=self._dtype, device=self._device)
        random_social_coefficent = torch.empty(
            1, dtype=self._dtype, device=self._device)

        nn.init.uniform_(random_cognitive_coefficent)
        nn.init.uniform_(random_social_coefficent)

        for name in self._velocity.keys():
            inertia = self._inertia_weight * self._velocity[name]
            cognitive = self._cognitive_coefficient * random_cognitive_coefficent * \
                (self._best_position[name] - self._current_position[name])
            social = self._social_coefficient * random_social_coefficent * \
                (self._global_best_position[name] -
                 self._current_position[name])

            self._velocity[name] = inertia + cognitive + social

            self._current_position[name] += self._learning_rate * \
                self._velocity[name]

    def step(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self._dtype is None and self._device is None:
            self._dtype = inputs.dtype
            self._device = inputs.device

        if self._best_loss is None and self._global_best_loss is None:
            self._initialize_best_results(inputs, target)

        self._calculate_new_position()

        step_best_loss = None

        for i in range(self._num_particles):
            state_dict = {name: parameters[i] for name,
                          parameters in self._current_position.items()}

            self._model.load_state_dict(state_dict)

            with torch.no_grad():
                outputs = self._model(inputs)

            loss = self._loss_function(outputs, target)

            if loss < self._best_loss[i]:
                for name in self._best_position:
                    self._best_position[name][i] = state_dict[name]

                self._best_loss[i] = loss

            if loss < self._global_best_loss:
                self._global_best_position = state_dict
                self._global_best_loss = loss

            if step_best_loss is not None:
                if loss < step_best_loss:
                    step_best_loss = loss
            else:
                step_best_loss = loss

        self._model.load_state_dict(self._global_best_position)

        return step_best_loss
