from roboflow import Roboflow
from ultralytics import YOLO

def main():
    rf = Roboflow(api_key="DPfbAOKvVuXUsAHW2dXZ")
    project = rf.workspace("iq-etg5a").project("my-first-project-z0sna")
    version = project.version(1)
    dataset = version.download("yolov11")
    
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        device=0,
        plots=True,
        batch=16,
        name="price_tag_model"
    )

if __name__ == "__main__":
    main()