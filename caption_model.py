from PIL import Image
import torch
import time
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

# here i am using blip + clip models for image captioning
class ImageCaptioner:
    def __init__(self, device=None):
        # using gpu if available
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.blip_name = "Salesforce/blip-image-captioning-base"
        self.blip_processor = BlipProcessor.from_pretrained(self.blip_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_name).to(self.device)

        self.clip_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_name).to(self.device)

        self.num_beams = 6
        self.num_return_sequences = 6
        self.max_new_tokens = 40

        print("[Captioner Ready — BLIP + CLIP loaded]")

    def _generate_candidates(self, image: Image):
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        output_ids = self.blip_model.generate(
            **inputs,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            max_new_tokens=self.max_new_tokens,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        captions = [self.blip_processor.decode(g, skip_special_tokens=True).strip() for g in output_ids]
        seen, unique = set(), []
        for c in captions:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)
        return unique

    def _clip_rerank(self, image: Image, candidates: list):
        clip_inputs = self.clip_processor(text=candidates, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            img_emb = self.clip_model.get_image_features(**{k: clip_inputs[k] for k in ["pixel_values"]})
            txt_emb = self.clip_model.get_text_features(input_ids=clip_inputs["input_ids"], attention_mask=clip_inputs["attention_mask"])
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sims = (txt_emb @ img_emb.T).squeeze().cpu().tolist()
        if isinstance(sims, float):
            sims = [sims]
        ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
        return ranked, int(torch.tensor(sims).argmax().item())

    def generate_caption(self, image_path, return_all=False):
        # here it is a main function for generating caption
        try:
            start = time.time()
            image = Image.open(image_path).convert("RGB")
            candidates = self._generate_candidates(image)
            ranked, best_idx = self._clip_rerank(image, candidates)
            best_caption = ranked[0][0] if ranked else (candidates[0] if candidates else "Unable to generate caption")
            print(f"[Captioning done in {time.time() - start:.2f}s]")
            if return_all:
                return best_caption, ranked
            return best_caption
        except Exception as e:
            print("[Error] Caption generation failed:", e)
            return "Error while generating caption."

# done by sarthak maddi for zidio internship
