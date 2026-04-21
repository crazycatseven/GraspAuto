"""Stage 3 contact-graph retrieval for the 2026-04-05 mainline.

Hybrid retrieval = learned similarity + graph mismatch + active mismatch
followed by top-k weighted Kabsch assembly and short rigid refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from graspauto.stage3_candidate_selector import LoadedCandidateSelector, load_candidate_selector
from graspauto.stage3_assembly import (
    apply_rigid,
    refine_rigid_pose,
    weighted_anchor_residual,
    weighted_kabsch,
)
from graspauto.stage3_contact_graph import DEFAULT_CODEBANK_METADATA
from graspauto.utils import FINGER_NAMES, resolve_device
from graspauto.stage3_utils import LightweightDiffusionSampler

EPS = 1e-8


@dataclass
class CandidateAssembly:
    code_id: torch.Tensor
    coarse_score: torch.Tensor
    rerank_score: torch.Tensor
    rot: torch.Tensor
    trans: torch.Tensor
    verts_world: torch.Tensor
    anchor_world: torch.Tensor
    anchor_rmse: torch.Tensor
    refine_loss: torch.Tensor
    active_mismatch: torch.Tensor
    center_residual: torch.Tensor
    palm_residual: torch.Tensor
    finger_overlap: torch.Tensor
    unified_overlap: torch.Tensor
    palm_overlap: torch.Tensor
    finger_overlap_max: torch.Tensor
    finger_overlap_min: torch.Tensor
    weights_mean: torch.Tensor
    weights_max: torch.Tensor
    weights_min: torch.Tensor
    pred_active_count: torch.Tensor
    code_active_count: torch.Tensor
    confident_fingers: torch.Tensor
    pairwise_dist_mismatch: torch.Tensor
    pairwise_dist_max: torch.Tensor
    pairwise_dist_top3_mean: torch.Tensor
    pairwise_dir_align: torch.Tensor
    pairwise_dir_min: torch.Tensor
    pairwise_dir_bottom3_mean: torch.Tensor
    active_subgraph_dist_mismatch: torch.Tensor
    active_subgraph_dir_bottom_mean: torch.Tensor


class Stage3ContactGraphRetriever:
    def __init__(
        self,
        metadata_path: Path | str = DEFAULT_CODEBANK_METADATA,
        device: str = "cpu",
        top_k: int = 16,
        active_threshold: float = 0.35,
        score_active_weight: float = 0.4,
        score_graph_weight: float = 0.5,
        score_dir_weight: float = 0.2,
        rerank_anchor_weight: float = 8.0,
        rerank_refine_weight: float = 4.0,
        rerank_overlap_weight: float = 2.0,
        rerank_palm_weight: float = 1.0,
        rerank_center_weight: float = 2.0,
        refine_steps: int = 10,
        refine_lr: float = 0.05,
        selector: LoadedCandidateSelector | None = None,
        selector_checkpoint: Path | str | None = None,
        use_learned_selector: bool = True,
        use_metric_retrieval: bool = False,
        diffusion_steps: int = 0,
        selector_rank_bias: Optional[Dict[int, float]] = None,
    ) -> None:
        self.device = resolve_device(device)
        self.top_k = int(top_k)
        self.active_threshold = float(active_threshold)
        self.score_active_weight = float(score_active_weight)
        self.score_graph_weight = float(score_graph_weight)
        self.score_dir_weight = float(score_dir_weight)
        self.rerank_anchor_weight = float(rerank_anchor_weight)
        self.rerank_refine_weight = float(rerank_refine_weight)
        self.rerank_overlap_weight = float(rerank_overlap_weight)
        self.rerank_palm_weight = float(rerank_palm_weight)
        self.rerank_center_weight = float(rerank_center_weight)
        self.refine_steps = int(refine_steps)
        self.refine_lr = float(refine_lr)
        self.use_learned_selector = bool(use_learned_selector)
        self.use_metric_retrieval = bool(use_metric_retrieval)
        self.diffusion_steps = int(diffusion_steps)
        if selector_rank_bias is None:
            selector_rank_bias = {4: 0.45, 6: 0.30}
        self.selector_rank_bias = {int(k): float(v) for k, v in selector_rank_bias.items()}
        self.diffusion_sampler = LightweightDiffusionSampler(num_steps=self.diffusion_steps) if self.diffusion_steps > 0 else None
        self.default_selector = selector
        if self.default_selector is None and self.use_learned_selector:
            self.default_selector = load_candidate_selector(
                checkpoint_path=selector_checkpoint,
                device=self.device,
                allow_missing=selector_checkpoint is None,
            )

        metadata = torch.load(Path(metadata_path), map_location="cpu", weights_only=False)
        self.pose_48 = metadata["pose_48"].to(self.device)
        self.mean_betas = metadata["mean_betas"].to(self.device)
        self.canonical_verts = metadata["canonical_verts"].to(self.device)
        self.canonical_anchors = metadata["canonical_anchors"].to(self.device)
        self.palm_center = metadata["palm_center"].to(self.device)
        self.palm_normal = metadata["palm_normal"].to(self.device)
        self.finger_dirs = metadata["finger_dirs"].to(self.device)
        self.active_finger_prob = metadata["active_finger_prob"].to(self.device)
        self.active_finger_mask = metadata["active_finger_mask"].to(self.device)
        self.finger_pairwise_dist = metadata["finger_pairwise_dist"].to(self.device)
        self.finger_pairwise_dir = metadata["finger_pairwise_dir"].to(self.device)
        self.code_feature = metadata["code_feature"].to(self.device)

    def _candidate_weights(
        self,
        active_prob: torch.Tensor,
        finger_mass: torch.Tensor,
        finger_entropy: torch.Tensor,
        code_active_prob: torch.Tensor,
        code_active_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_mask = active_prob >= self.active_threshold
        weights = active_prob.clamp_min(0.05)
        weights = weights * (finger_mass / finger_mass.max().clamp_min(EPS)).clamp_min(0.05)
        weights = weights * (1.0 / (1.0 + finger_entropy)).clamp_min(0.05)
        weights = weights * code_active_prob.clamp_min(0.05)
        shared = pred_mask & code_active_mask
        if bool(shared.any().item()):
            weights = weights * shared.float()
        if float(weights.sum().item()) < EPS:
            weights = active_prob.clamp_min(0.05) * code_active_prob.clamp_min(0.05)
        if float(weights.sum().item()) < EPS:
            weights = code_active_mask.float().clamp_min(0.05)
        if float(weights.sum().item()) < EPS:
            weights = torch.ones_like(active_prob)
        return weights

    def coarse_scores(self, model_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_metric_retrieval and "query_embed" in model_out and "code_metric_embed" in model_out:
            query_embed = model_out["query_embed"].to(self.device)
            code_metric_embed = model_out["code_metric_embed"].to(self.device)
            metric_scores = torch.matmul(query_embed, code_metric_embed.t())
            scale = model_out.get("logit_scale_value")
            if scale is not None:
                metric_scores = metric_scores * scale.to(self.device)
            if self.diffusion_sampler is not None and metric_scores.shape[-1] > self.top_k:
                prefetch_k = min(max(self.top_k * 2, self.top_k), metric_scores.shape[-1])
                top_vals, top_idx = metric_scores.topk(k=prefetch_k, dim=-1)
                candidate_embed = code_metric_embed.index_select(0, top_idx.reshape(-1)).reshape(query_embed.shape[0], prefetch_k, -1)
                refined_query = self.diffusion_sampler(query_embed, candidate_embed)
                refined_scores = torch.einsum("bd,bkd->bk", refined_query, candidate_embed)
                if scale is not None:
                    refined_scores = refined_scores * scale.to(self.device)
                metric_scores = metric_scores.clone()
                metric_scores.scatter_(1, top_idx, refined_scores)
            return metric_scores

        code_logits = model_out["code_logits"].to(self.device)
        active_prob = model_out["active_finger_prob"].to(self.device)
        graph = model_out["graph"]
        pred_dist = graph["finger_pairwise_dist"].to(self.device)
        pred_dirs = graph["finger_dirs_from_palm"].to(self.device)

        active_mismatch = (active_prob.unsqueeze(1) - self.active_finger_prob.unsqueeze(0)).abs().mean(dim=-1)
        graph_dist_mismatch = (pred_dist.unsqueeze(1) - self.finger_pairwise_dist.unsqueeze(0)).abs().mean(dim=(-1, -2))
        dir_mismatch = 1.0 - (pred_dirs.unsqueeze(1) * self.finger_dirs.unsqueeze(0)).sum(dim=-1).mean(dim=-1)

        return (
            code_logits
            - self.score_active_weight * active_mismatch
            - self.score_graph_weight * graph_dist_mismatch
            - self.score_dir_weight * dir_mismatch
        )

    def _assemble_one(
        self,
        sample_idx: int,
        batch: Dict[str, torch.Tensor],
        model_out: Dict[str, torch.Tensor],
        code_id: int,
        coarse_score: torch.Tensor,
    ) -> CandidateAssembly:
        code_id_t = torch.tensor(code_id, device=self.device, dtype=torch.long)
        code_anchor = self.canonical_anchors[code_id_t]
        code_palm = self.palm_center[code_id_t]
        code_active_prob = self.active_finger_prob[code_id_t]
        code_active_mask = self.active_finger_mask[code_id_t] > 0.5

        graph = model_out["graph"]
        pred_centers = graph["finger_centroid"][sample_idx].to(self.device)
        pred_palm = graph["palm_centroid"][sample_idx].to(self.device)
        pred_mass = graph["finger_mass"][sample_idx].to(self.device)
        pred_entropy = graph["finger_entropy"][sample_idx].to(self.device)
        active_prob = model_out["active_finger_prob"][sample_idx].to(self.device)
        unified_prob = model_out["unified_contact_prob"][sample_idx].to(self.device)
        finger_prob = model_out["finger_contact_prob"][sample_idx].to(self.device)
        palm_prob = model_out["palm_contact_prob"][sample_idx].to(self.device)
        object_points = batch["object_points"][sample_idx].to(self.device)

        weights = self._candidate_weights(active_prob, pred_mass, pred_entropy, code_active_prob, code_active_mask)
        pred_active_mask = active_prob >= self.active_threshold
        confident_fingers = int((pred_active_mask | code_active_mask).sum().item())
        palm_weight = weights.mean() * (0.35 if confident_fingers < 3 else 0.15)

        source_points = torch.cat([code_anchor, code_palm.unsqueeze(0)], dim=0)
        target_points = torch.cat([pred_centers, pred_palm.unsqueeze(0)], dim=0)
        full_weights = torch.cat([weights, palm_weight.unsqueeze(0)], dim=0)

        rigid = weighted_kabsch(source_points, target_points, full_weights)
        anchor_world = apply_rigid(code_anchor, rigid.rot, rigid.trans)
        anchor_residual = weighted_anchor_residual(code_anchor, pred_centers, weights, rigid.rot, rigid.trans)

        refine = refine_rigid_pose(
            source_points=source_points,
            target_points=target_points,
            weights=full_weights,
            init_rot=rigid.rot,
            init_trans=rigid.trans,
            steps=self.refine_steps,
            lr=self.refine_lr,
        )
        anchor_world_refined = apply_rigid(code_anchor, refine.rot, refine.trans)
        palm_world_refined = apply_rigid(code_palm.unsqueeze(0), refine.rot, refine.trans).squeeze(0)
        anchor_residual_refined = weighted_anchor_residual(code_anchor, pred_centers, weights, refine.rot, refine.trans)
        palm_residual_refined = (palm_world_refined - pred_palm).norm()

        dist_to_points = torch.cdist(anchor_world_refined.unsqueeze(0), object_points.unsqueeze(0)).squeeze(0)
        nn_idx = dist_to_points.argmin(dim=1)
        finger_overlap_vec = finger_prob[nn_idx, torch.arange(len(FINGER_NAMES), device=self.device)]
        finger_overlap = (finger_overlap_vec * weights).sum() / weights.sum().clamp_min(EPS)
        unified_overlap = unified_prob[nn_idx].mean()
        palm_nn_idx = (object_points - palm_world_refined.unsqueeze(0)).norm(dim=-1).argmin()
        palm_overlap = 0.5 * palm_prob[palm_nn_idx] + 0.5 * unified_prob[palm_nn_idx]
        center_residual = ((anchor_world_refined - pred_centers).norm(dim=-1) * weights).sum() / weights.sum().clamp_min(EPS)

        active_mismatch = F.binary_cross_entropy(
            active_prob.clamp(1e-4, 1.0 - 1e-4),
            code_active_prob.clamp(1e-4, 1.0 - 1e-4),
        )
        pred_pairwise_dist = graph["finger_pairwise_dist"][sample_idx].to(self.device)
        code_pairwise_dist = self.finger_pairwise_dist[code_id_t]
        pairwise_dist_abs = (pred_pairwise_dist - code_pairwise_dist).abs()
        pairwise_dist_flat = pairwise_dist_abs.reshape(-1)
        pairwise_dist_mismatch = pairwise_dist_flat.mean()
        pairwise_dist_max = pairwise_dist_flat.max()
        pairwise_dist_top3_mean = pairwise_dist_flat.topk(k=min(3, pairwise_dist_flat.numel())).values.mean()
        pred_dirs = graph["finger_dirs_from_palm"][sample_idx].to(self.device)
        code_dirs = self.finger_dirs[code_id_t]
        pairwise_dir_cos = (pred_dirs * code_dirs).sum(dim=-1)
        pairwise_dir_flat = pairwise_dir_cos.reshape(-1)
        pairwise_dir_align = pairwise_dir_flat.mean()
        pairwise_dir_min = pairwise_dir_flat.min()
        pairwise_dir_bottom3_mean = pairwise_dir_flat.topk(k=min(3, pairwise_dir_flat.numel()), largest=False).values.mean()
        active_union = (pred_active_mask | code_active_mask).float()
        active_pair_mask = torch.outer(active_union, active_union)
        active_pair_mask.fill_diagonal_(0.0)
        active_pair_denom = active_pair_mask.sum().clamp_min(1.0)
        active_subgraph_dist_mismatch = (pairwise_dist_abs * active_pair_mask).sum() / active_pair_denom
        active_subgraph_dir_bottom_mean = (pairwise_dir_cos * active_pair_mask + (1.0 - active_pair_mask)).reshape(-1)
        active_subgraph_dir_bottom_mean = active_subgraph_dir_bottom_mean.topk(k=min(3, int(active_pair_mask.sum().item()) if int(active_pair_mask.sum().item()) > 0 else 1), largest=False).values.mean()
        rerank_score = (
            coarse_score
            - self.rerank_anchor_weight * anchor_residual_refined
            - self.rerank_refine_weight * active_mismatch
            - self.rerank_center_weight * center_residual
            - self.rerank_palm_weight * palm_residual_refined
            + self.rerank_overlap_weight * finger_overlap
            + 0.5 * self.rerank_overlap_weight * unified_overlap
            + 0.5 * self.rerank_palm_weight * palm_overlap
        )
        verts_world = apply_rigid(self.canonical_verts[code_id_t], refine.rot, refine.trans)

        return CandidateAssembly(
            code_id=code_id_t,
            coarse_score=coarse_score,
            rerank_score=rerank_score,
            rot=refine.rot,
            trans=refine.trans,
            verts_world=verts_world,
            anchor_world=anchor_world_refined,
            anchor_rmse=anchor_residual_refined,
            refine_loss=refine.loss,
            active_mismatch=active_mismatch.detach(),
            center_residual=center_residual.detach(),
            palm_residual=palm_residual_refined.detach(),
            finger_overlap=finger_overlap.detach(),
            unified_overlap=unified_overlap.detach(),
            palm_overlap=palm_overlap.detach(),
            finger_overlap_max=finger_overlap_vec.max().detach(),
            finger_overlap_min=finger_overlap_vec.min().detach(),
            weights_mean=weights.mean().detach(),
            weights_max=weights.max().detach(),
            weights_min=weights.min().detach(),
            pred_active_count=pred_active_mask.sum().detach().to(dtype=torch.float32),
            code_active_count=code_active_mask.sum().detach().to(dtype=torch.float32),
            confident_fingers=torch.tensor(float(confident_fingers), device=self.device),
            pairwise_dist_mismatch=pairwise_dist_mismatch.detach(),
            pairwise_dist_max=pairwise_dist_max.detach(),
            pairwise_dist_top3_mean=pairwise_dist_top3_mean.detach(),
            pairwise_dir_align=pairwise_dir_align.detach(),
            pairwise_dir_min=pairwise_dir_min.detach(),
            pairwise_dir_bottom3_mean=pairwise_dir_bottom3_mean.detach(),
            active_subgraph_dist_mismatch=active_subgraph_dist_mismatch.detach(),
            active_subgraph_dir_bottom_mean=active_subgraph_dir_bottom_mean.detach(),
        )

    def assemble_topk_candidates(
        self,
        batch: Dict[str, torch.Tensor],
        model_out: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[CandidateAssembly]]]:
        coarse = self.coarse_scores(model_out)
        topk = coarse.topk(k=min(self.top_k, coarse.shape[-1]), dim=-1)
        candidate_ids = topk.indices
        candidate_scores = topk.values

        candidate_assemblies: List[List[CandidateAssembly]] = []
        for batch_idx in range(candidate_ids.shape[0]):
            assemblies = []
            for rank in range(candidate_ids.shape[1]):
                code_id = int(candidate_ids[batch_idx, rank].item())
                assemblies.append(
                    self._assemble_one(
                        sample_idx=batch_idx,
                        batch=batch,
                        model_out=model_out,
                        code_id=code_id,
                        coarse_score=candidate_scores[batch_idx, rank],
                    )
                )
            candidate_assemblies.append(assemblies)
        return coarse, candidate_ids, candidate_scores, candidate_assemblies

    def _pack_selected_candidates(
        self,
        coarse: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_assemblies: List[List[CandidateAssembly]],
        selected_rank: List[int],
        selection_scores: List[torch.Tensor],
        selector_logits: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        best_candidates = [assemblies[rank] for assemblies, rank in zip(candidate_assemblies, selected_rank)]
        payload = {
            "best_code_id": torch.stack([item.code_id for item in best_candidates], dim=0),
            "best_score": torch.stack([item.rerank_score for item in best_candidates], dim=0),
            "selection_score": torch.stack(selection_scores, dim=0),
            "best_rerank_score": torch.stack([item.rerank_score for item in best_candidates], dim=0),
            "world_rot": torch.stack([item.rot for item in best_candidates], dim=0),
            "world_trans": torch.stack([item.trans for item in best_candidates], dim=0),
            "verts_world": torch.stack([item.verts_world for item in best_candidates], dim=0),
            "anchor_world": torch.stack([item.anchor_world for item in best_candidates], dim=0),
            "anchor_rmse": torch.stack([item.anchor_rmse for item in best_candidates], dim=0),
            "refine_loss": torch.stack([item.refine_loss for item in best_candidates], dim=0),
            "active_mismatch": torch.stack([item.active_mismatch for item in best_candidates], dim=0),
            "center_residual": torch.stack([item.center_residual for item in best_candidates], dim=0),
            "palm_residual": torch.stack([item.palm_residual for item in best_candidates], dim=0),
            "finger_overlap": torch.stack([item.finger_overlap for item in best_candidates], dim=0),
            "unified_overlap": torch.stack([item.unified_overlap for item in best_candidates], dim=0),
            "palm_overlap": torch.stack([item.palm_overlap for item in best_candidates], dim=0),
            "candidate_ids": candidate_ids,
            "candidate_scores": candidate_scores,
            "candidate_rerank_scores": torch.stack(
                [torch.stack([item.rerank_score for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "candidate_anchor_rmse": torch.stack(
                [torch.stack([item.anchor_rmse for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "candidate_refine_loss": torch.stack(
                [torch.stack([item.refine_loss for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "candidate_active_mismatch": torch.stack(
                [torch.stack([item.active_mismatch for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "candidate_center_residual": torch.stack(
                [torch.stack([item.center_residual for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "candidate_palm_residual": torch.stack(
                [torch.stack([item.palm_residual for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "candidate_finger_overlap": torch.stack(
                [torch.stack([item.finger_overlap for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "candidate_unified_overlap": torch.stack(
                [torch.stack([item.unified_overlap for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "candidate_palm_overlap": torch.stack(
                [torch.stack([item.palm_overlap for item in assemblies], dim=0) for assemblies in candidate_assemblies],
                dim=0,
            ),
            "selected_rank": torch.tensor(selected_rank, device=self.device, dtype=torch.long),
            "coarse_scores": coarse,
        }
        if selector_logits is not None:
            payload["selector_logits"] = torch.stack(selector_logits, dim=0)
        return payload

    def _resolve_selector(
        self,
        selector: LoadedCandidateSelector | None = None,
        use_learned_selector: bool | None = None,
    ) -> LoadedCandidateSelector | None:
        if selector is not None:
            return selector
        if use_learned_selector is False:
            return None
        if use_learned_selector is True or self.use_learned_selector:
            return self.default_selector
        return None

    def select_from_candidates(
        self,
        coarse: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_assemblies: List[List[CandidateAssembly]],
        selector: LoadedCandidateSelector | None = None,
        use_learned_selector: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        effective_selector = self._resolve_selector(selector=selector, use_learned_selector=use_learned_selector)
        selected_rank: List[int] = []
        selection_scores: List[torch.Tensor] = []
        selector_logits: List[torch.Tensor] = []

        for assemblies in candidate_assemblies:
            if effective_selector is None:
                best_idx = max(range(len(assemblies)), key=lambda idx: float(assemblies[idx].rerank_score.item()))
                selected_rank.append(best_idx)
                selection_scores.append(assemblies[best_idx].rerank_score)
                continue

            best_idx, logits = effective_selector.select_assembly(assemblies)
            if self.selector_rank_bias:
                adjusted_logits = logits.clone()
                for rank_idx, penalty in self.selector_rank_bias.items():
                    if 0 <= rank_idx < adjusted_logits.shape[0]:
                        adjusted_logits[rank_idx] = adjusted_logits[rank_idx] - penalty
                logits = adjusted_logits
                best_idx = int(logits.argmax().item())
            selected_rank.append(best_idx)
            selection_scores.append(logits[best_idx])
            selector_logits.append(logits)

        return self._pack_selected_candidates(
            coarse=coarse,
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            candidate_assemblies=candidate_assemblies,
            selected_rank=selected_rank,
            selection_scores=selection_scores,
            selector_logits=selector_logits if effective_selector is not None else None,
        )

    def decode_batch(
        self,
        batch: Dict[str, torch.Tensor],
        model_out: Dict[str, torch.Tensor],
        selector: LoadedCandidateSelector | None = None,
        use_learned_selector: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.select_from_candidates(
            *self.assemble_topk_candidates(batch, model_out),
            selector=selector,
            use_learned_selector=use_learned_selector,
        )

    def decode_batch_with_selector(
        self,
        batch: Dict[str, torch.Tensor],
        model_out: Dict[str, torch.Tensor],
        selector: LoadedCandidateSelector,
    ) -> Dict[str, torch.Tensor]:
        return self.decode_batch(batch, model_out, selector=selector, use_learned_selector=True)


__all__ = ["Stage3ContactGraphRetriever", "CandidateAssembly"]
