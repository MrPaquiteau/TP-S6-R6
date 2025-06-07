import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt

ToSpectrogram = torchaudio.transforms.MelSpectrogram()
ToDB = torchaudio.transforms.AmplitudeToDB()

def main():
    # Paramètres d'affiche
    parser = argparse.ArgumentParser(description='Affichage de fichiers sonores')
    parser.add_argument('filename')
    parser.add_argument('--spectrogram', action='store_true')
    parser.set_defaults(spectrogram=False)

    args = parser.parse_args()
    print( "Read '{}' with option spectrogram={}".format(args.filename, args.spectrogram) )

    # chargement de l'audio
    waveform, sample_rate = torchaudio.load(args.filename)

    # Show Audio with or without spectrogram
    if args.spectrogram:
        spectrogram = ToSpectrogram(waveform)
        mel_spectrogram = ToDB(spectrogram)
        fig = plt.figure(figsize=(12, 8))
        
        fig.add_subplot(211).set_title('Audio')
        plt.plot(waveform.t().numpy())  # Affiche l'onde audio

        fig.add_subplot(212).set_title('MelSpectrogram')
        plt.imshow(mel_spectrogram[0].numpy(), cmap='inferno', origin='lower', aspect='auto')
        plt.ylabel("Mel Frequency Bin")
        plt.xlabel("Time Frame")
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(waveform.numpy())
    
    plt.show()
    
    print(f"Sample rate: {sample_rate} Hz") # Taux d'échantillonnage de l'audio, le nombre d'échantillons audio enregistrés par seconde.
    print(f"Waveform shape: {waveform.shape}") # forme de l'onde, (nombre de canals audio, le temps)
    print(f"Spectrogram shape: {spectrogram.shape}") # forme du spectrogramme (son représenté en fréquence en fonction du temps), (nombre de bande de fréquences, nombre de fenetres temporelles)
    print(f"Duration: {waveform.shape[1]/sample_rate:.2f} seconds") # Durée de l'extrait, nombre d'échantillons / taux d'échantillonnage

if __name__ == '__main__':
    main()