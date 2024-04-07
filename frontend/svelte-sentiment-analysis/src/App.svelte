<script>
    import { writable } from 'svelte/store';

    // Using a store for reactive updates
    let review = '';
    const sentiment = writable('');

    async function analyzeSentiment() {
        // Provide immediate feedback that the analysis is in progress
        sentiment.set('Analyzing...');

        try {
            const response = await fetch('4.157.184.66:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*', // For development. Adjust for production
                },
                body: JSON.stringify({ reviews: [review] }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            sentiment.set(result.predictions[0]); // Update sentiment with the result
        } catch (error) {
            console.error('Error:', error);
            sentiment.set('Error analyzing sentiment. Please try again.');
        }

        // Optionally clear the review or leave it for user reference
        // review = '';
    }
</script>

<style>
    /* Additional styling for better appearance */
    main {
        font-family: Arial, sans-serif;
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
    }
    textarea {
        width: 100%;
        height: 100px;
        margin-bottom: 20px;
        padding: 10px;
        font-size: 16px;
        box-sizing: border-box;
    }
    button {
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
</style>

<main>
    <h1>Sentiment Analysis</h1>
    <textarea bind:value={review} placeholder="Enter your review"></textarea>
    <button on:click={analyzeSentiment}>Analyze</button>
    <p>Sentiment: {$sentiment}</p> <!-- Reactive sentiment display -->
</main>
