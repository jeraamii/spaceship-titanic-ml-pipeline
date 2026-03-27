from src.pipeline.pipeline import run_pipeline


def main():
    print("Starting training pipeline...\n")
    
    run_pipeline()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()