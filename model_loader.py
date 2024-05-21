from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

from PIL import Image
import requests
import torch


class model_loader():
    def __init__(self) -> None:
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
            cache_dir="PATH/TO/CACHE/DIR"  # Defaults to ~/.cache
        )
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        pass
    

    def load_pretrained_model(self):
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)

    def demo1(self):
        image_processor = self.image_processor
        tokenizer = self.tokenizer
        model = self.model

        """
        Step 1: Load images
        """
        demo_image_one = Image.open(
            requests.get(
                "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
            ).raw
        )

        demo_image_two = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
                stream=True
            ).raw
        )

        query_image = Image.open(
            requests.get(
                "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
                stream=True
            ).raw
        )        

        """
        Step 2: Preprocessing images
        Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
        batch_size x num_media x num_frames x channels x height x width. 
        In this case batch_size = 1, num_media = 3, num_frames = 1,
        channels = 3, height = 224, width = 224.
        """
        vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        """
        Step 3: Preprocessing text
        Details: In the text we expect an <image> special token to indicate where an image is.
        We also expect an <|endofchunk|> special token to indicate the end of the text 
        portion associated with an image.
        """
        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = tokenizer(
            ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
            return_tensors="pt",
        )


        """
        Step 4: Generate text
        """
        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )

        print("Generated text: ", tokenizer.decode(generated_text[0]))

if __name__ == "__main__":
    print("copy some code and run a demo of openflamingo")
    flag = 0 
    shishi_model = model_loader()
    shishi_model.load_pretrained_model()
    if flag==0:
        shishi_model.demo1()