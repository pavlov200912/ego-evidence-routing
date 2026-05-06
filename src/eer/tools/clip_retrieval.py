"""CLIP/SigLIP text-image retrieval tool."""

from __future__ import annotations

import logging

import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel
import heapq

from eer.data.frames import Frame
from eer.tools.base import EvidenceTool

logger = logging.getLogger(__name__)


class CLIPRetrievalTool(EvidenceTool):
    """Select frames most semantically relevant to the question via CLIP.

    Embeds all candidate frame images and the question text, then returns
    the top-K frames by cosine similarity and time variation.
    """
    fm: FrameExtractorClip
    def __init__(
        self,
        model_name: str = "ViT-SO400M-14-SigLIP",
        pretrained: str = "webli",
        device: str | None = None,
    ) -> None:
        """Load the open_clip model and preprocessing transforms.

        Args:
            model_name: open_clip architecture name.
            pretrained: Pretrained weights tag.
            device: Torch device string; defaults to CUDA if available.
        """
        

        super().__init__()
        self.fm = FrameExtractorClip("clip",device = device)

        

    @property
    def name(self) -> str:
        return "clip"

    
    def select(
        self,
        candidate_frames: list[Frame],
        question: str,
        budget: int = 8,
        **_kwargs,
    ) -> list[Frame]:
        """Return top-*budget* frames by cosine similarity to *question*.

        Args:
            candidate_frames: All available frames for the clip.
            question: VQA question used to construct the text query.
            budget: Number of frames to return.

        Returns:
            Selected frames sorted by ascending timestamp.
        """
        if not candidate_frames:
            return []

        if len(candidate_frames) <= budget:
            return list(candidate_frames)
        
        
        scores,frame_num=self.fm.give_scores(candidate_frames, question)
        outs, _ = self.fm.select_frames(scores, frame_num, max_num_frames=budget)
        selected = [candidate_frames[i] for i in outs[0]]
        logger.debug(
            "CLIPRetrievalTool: selected %d/%d frames", len(selected), len(candidate_frames)
        )
        return selected


class FrameExtractorClip:
    model_name = None
    model = None
    vis_processors = None
    text_processors = None
    processor = None
    device = None

    def __init__(self, model_name, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        #if model_name == 'blip':
            #self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
        if model_name == 'clip':
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise ValueError("model not support")
        
    def give_scores(self, frames: list[Frame], question: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract frames and calculate the similarity scores between the frames and the question using time variation as well.
        takes around 0.25s per frame for clip
        args:
            frames: the frames to extract and calculate scores for
            question: the question to calculate the similarity score
            processing_fps: the fps to process the video, default is 1, which means only calculate the similarity score for one frame per second
        return:
            score: the similarity scores between the frames and the question
            frame_num: the frame numbers of the extracted frames
        """

        text = question
        
        frame_nums = (len(frames))

        score = []
        frame_num = []

        
        
        if self.model_name == 'clip':
            inputs_text = self.processor(text=text, return_tensors="pt", padding=True,truncation=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs_text)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            for j in range(frame_nums):
                raw_image = frames[j].image
                inputs_image = self.processor(images=raw_image, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs_image)
                if not isinstance(image_features, torch.Tensor):
                    image_features = image_features.pooler_output
                clip_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
                score.append(clip_score.item())
                frame_num.append(j)

        else:
            raise ValueError("model not support")
        
        return np.array(score), np.array(frame_num)
    
    
    def meanstd(self, len_scores, dic_scores, n, fns, t1, t2, all_depth):
        split_scores = []
        split_fn = []
        no_split_scores = []
        no_split_fn = []
        i= 0
        for dic_score, fn in zip(dic_scores, fns):
                # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
                score = dic_score['score']
                depth = dic_score['depth']
                mean = np.mean(score)
                std = np.std(score)

                top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
                top_score = [score[t] for t in top_n]
                # print(f"split {i}: ",len(score))
                i += 1
                mean_diff = np.mean(top_score) - mean
                if mean_diff > t1 and std > t2:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
                elif depth < all_depth:
                # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
                        score1 = score[:len(score)//2]
                        score2 = score[len(score)//2:]
                        fn1 = fn[:len(score)//2]
                        fn2 = fn[len(score)//2:]                       
                        split_scores.append(dict(score=score1,depth=depth+1))
                        split_scores.append(dict(score=score2,depth=depth+1))
                        split_fn.append(fn1)
                        split_fn.append(fn2)
                else:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
        if len(split_scores) > 0:
                all_split_score, all_split_fn = self.meanstd(len_scores, split_scores, n, split_fn,t1,t2,all_depth)
        else:
                all_split_score = []
                all_split_fn = []
        all_split_score = no_split_scores + all_split_score
        all_split_fn = no_split_fn + all_split_fn


        return all_split_score, all_split_fn

    def select_frames(self,scores, frame_num,max_num_frames=64, ratio=1, t1 = 0.8, t2 = -100, all_depth = 5):
        """
        Select frames based on the similarity scores and the mean and std of the scores.
        args:
            scores: the similarity scores between the frames and the question
            frame_num: the frame numbers of the extracted frames
            n: the number of frames to select, default is 8
            t1: the threshold for the mean difference between the top n scores and the mean score, default is 0.8
            t2: the threshold for the std of the scores, default is -100
            all_depth: the maximum depth of splitting, default is 5
        return:
            selected_frames: the selected frame numbers
        """
        outs = []
        segs = []
        itm_out = scores
        fn_out = frame_num

        
        nums = int(len(itm_out)/ratio)
        new_score = [itm_out[num*ratio] for num in range(nums)]
        new_fnum = [fn_out[num*ratio] for num in range(nums)]
        score = np.array(new_score)
        fn = new_fnum
        num = max_num_frames
        if len(score) >= num:
            normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
            a, b = self.meanstd(len(score), [dict(score=normalized_data,depth=0)], num, [fn], t1, t2, all_depth)
            segs.append(len(a))
            out = []
            if len(score) >= num:
                for s,f in zip(a,b):
                    f_num = max(1, int(num / 2**(s['depth'])))
                    topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                    f_nums = [f[t] for t in topk]
                    out.extend(f_nums)
            # pick top-num by CLIP score, then restore temporal order
            out = sorted(set(out), key=lambda i: score[i], reverse=True)[:num]
            out.sort()
            outs.append(out)
        else:
            outs.append(fn)
        return outs, segs
        
    