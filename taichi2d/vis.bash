for epoch in {1..30}; do
    python compare_samples.py --dataset dataset_region --epoch $epoch
done

for epoch in {1..30}; do
    python compare_samples.py --dataset dataset_uni --epoch $epoch
done


