import fetch from "node-fetch";
import fs from 'graceful-fs';
import { gotScraping } from 'got-scraping';
import { stringify } from 'csv-stringify/sync';

(async () => {
    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    const fetchTweets = async (url, retries = 7) => {
        for (let i = 0; i < retries; i++) {
            try {
                const res = await fetch(url, {
                    headers: {
                        // PASTE OWN HEADERS HERE
                    },
                    body: null,
                    method: "GET"
                });
                const json = await res.json();
                fs.writeFileSync("test.json", JSON.stringify(json, null, 2))
                return json;
            } catch (error) {
                console.error(`Attempt ${i + 1} failed:`, error.message);
                if (i === retries - 1) throw error;
                console.log(`Retrying in 10 minutes...`);
                await delay(603000); // Wait for 10 minutes before retrying
            }
        }
    };

    const decodeHtmlEntities = (text) => {
        return text
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&le;/g, '≤')
            .replace(/&ge;/g, '≥');
    };

    const extractTweets = (data, startIndex, endIndex) => {
        let tweets = [];
        for (let i = startIndex; i < endIndex; i++) {
            let tweet = data.data.search_by_raw_query.search_timeline.timeline.instructions[0].entries[i];
            if (tweet.content && tweet.content.itemContent && tweet.content.itemContent.tweet_results && tweet.content.itemContent.tweet_results.result && tweet.content.itemContent.tweet_results.result.legacy) {
                let tweetText = tweet.content.itemContent.tweet_results.result.legacy.full_text;
                let tweetDate = tweet.content.itemContent.tweet_results.result.legacy.created_at;

                // Remove links starting with "https://t.co/"
                tweetText = tweetText.replace(/https:\/\/t\.co\/\S+/g, '');

                // Replace newline characters with spaces
                tweetText = tweetText.replace(/\n/g, ' ');

                // Decode HTML entities
                tweetText = decodeHtmlEntities(tweetText);

                tweets.push({ date: tweetDate, text: tweetText });
            }
        }
        return tweets;
    };

    const processTweets = async (numRequests) => {
        let initialUrl = "https://x.com/i/api/graphql/6uoFezW1o4e-n-VI5vfksA/SearchTimeline?variables=%7B%22rawQuery%22%3A%22Stock%20StockMarket%20Price%20%20(SPY%20OR%20QQQ%20OR%20ARKK%20OR%20SMH%20OR%20AAPL%20OR%20NFLX%20OR%20TSLA%20OR%20META%20OR%20AMZN%20OR%20NVDA%20OR%20GOOG%20OR%20MSFT%20OR%20SHOP%20OR%20AMD%20OR%20UPST%20OR%20AAL%20OR%20TSM%20OR%20Nasdaq%20OR%20Dow)%20lang%3Aen%20until%3A2024-07-01%20since%3A2024-06-01%22%2C%22count%22%3A20%2C%22querySource%22%3A%22typed_query%22%2C%22product%22%3A%22Top%22%7D&features=%7B%22rweb_tipjar_consumption_enabled%22%3Atrue%2C%22responsive_web_graphql_exclude_directive_enabled%22%3Atrue%2C%22verified_phone_label_enabled%22%3Afalse%2C%22creator_subscriptions_tweet_preview_api_enabled%22%3Atrue%2C%22responsive_web_graphql_timeline_navigation_enabled%22%3Atrue%2C%22responsive_web_graphql_skip_user_profile_image_extensions_enabled%22%3Afalse%2C%22communities_web_enable_tweet_community_results_fetch%22%3Atrue%2C%22c9s_tweet_anatomy_moderator_badge_enabled%22%3Atrue%2C%22articles_preview_enabled%22%3Atrue%2C%22tweetypie_unmention_optimization_enabled%22%3Atrue%2C%22responsive_web_edit_tweet_api_enabled%22%3Atrue%2C%22graphql_is_translatable_rweb_tweet_is_translatable_enabled%22%3Atrue%2C%22view_counts_everywhere_api_enabled%22%3Atrue%2C%22longform_notetweets_consumption_enabled%22%3Atrue%2C%22responsive_web_twitter_article_tweet_consumption_enabled%22%3Atrue%2C%22tweet_awards_web_tipping_enabled%22%3Afalse%2C%22creator_subscriptions_quote_tweet_preview_enabled%22%3Afalse%2C%22freedom_of_speech_not_reach_fetch_enabled%22%3Atrue%2C%22standardized_nudges_misinfo%22%3Atrue%2C%22tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled%22%3Atrue%2C%22rweb_video_timestamps_enabled%22%3Atrue%2C%22longform_notetweets_rich_text_read_enabled%22%3Atrue%2C%22longform_notetweets_inline_media_enabled%22%3Atrue%2C%22responsive_web_enhance_cards_enabled%22%3Afalse%7D";

        let url = initialUrl;

        for (let i = 0; i < numRequests; i++) {
            try {
                let data = await fetchTweets(url);

                const entries = data?.data?.search_by_raw_query?.search_timeline?.timeline?.instructions[0]?.entries;
                const entries_length = entries.length;
                console.log("Entries found: ", entries.length);

                let tweets;
                if (i === 0) {
                    // First request
                    tweets = extractTweets(data, 1, 20);
                    url = `https://x.com/i/api/graphql/6uoFezW1o4e-n-VI5vfksA/SearchTimeline?variables=%7B%22rawQuery%22%3A%22Stock%20StockMarket%20Price%20%20(SPY%20OR%20QQQ%20OR%20ARKK%20OR%20SMH%20OR%20AAPL%20OR%20NFLX%20OR%20TSLA%20OR%20META%20OR%20AMZN%20OR%20NVDA%20OR%20GOOG%20OR%20MSFT%20OR%20SHOP%20OR%20AMD%20OR%20UPST%20OR%20AAL%20OR%20TSM%20OR%20Nasdaq%20OR%20Dow)%20lang%3Aen%20until%3A2024-07-01%20since%3A2024-06-01%22%2C%22count%22%3A20%2C%22cursor%22%3A%22${data.data.search_by_raw_query.search_timeline.timeline.instructions[0].entries[21].content.value}%22%2C%22querySource%22%3A%22typed_query%22%2C%22product%22%3A%22Top%22%7D&features=%7B%22rweb_tipjar_consumption_enabled%22%3Atrue%2C%22responsive_web_graphql_exclude_directive_enabled%22%3Atrue%2C%22verified_phone_label_enabled%22%3Afalse%2C%22creator_subscriptions_tweet_preview_api_enabled%22%3Atrue%2C%22responsive_web_graphql_timeline_navigation_enabled%22%3Atrue%2C%22responsive_web_graphql_skip_user_profile_image_extensions_enabled%22%3Afalse%2C%22communities_web_enable_tweet_community_results_fetch%22%3Atrue%2C%22c9s_tweet_anatomy_moderator_badge_enabled%22%3Atrue%2C%22articles_preview_enabled%22%3Atrue%2C%22tweetypie_unmention_optimization_enabled%22%3Atrue%2C%22responsive_web_edit_tweet_api_enabled%22%3Atrue%2C%22graphql_is_translatable_rweb_tweet_is_translatable_enabled%22%3Atrue%2C%22view_counts_everywhere_api_enabled%22%3Atrue%2C%22longform_notetweets_consumption_enabled%22%3Atrue%2C%22responsive_web_twitter_article_tweet_consumption_enabled%22%3Atrue%2C%22tweet_awards_web_tipping_enabled%22%3Afalse%2C%22creator_subscriptions_quote_tweet_preview_enabled%22%3Afalse%2C%22freedom_of_speech_not_reach_fetch_enabled%22%3Atrue%2C%22standardized_nudges_misinfo%22%3Atrue%2C%22tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled%22%3Atrue%2C%22rweb_video_timestamps_enabled%22%3Atrue%2C%22longform_notetweets_rich_text_read_enabled%22%3Atrue%2C%22longform_notetweets_inline_media_enabled%22%3Atrue%2C%22responsive_web_enhance_cards_enabled%22%3Afalse%7D`;
                } else {
                    // Subsequent requests
                    tweets = extractTweets(data, 0, entries_length);
                    url = `https://x.com/i/api/graphql/6uoFezW1o4e-n-VI5vfksA/SearchTimeline?variables=%7B%22rawQuery%22%3A%22Stock%20StockMarket%20Price%20%20(SPY%20OR%20QQQ%20OR%20ARKK%20OR%20SMH%20OR%20AAPL%20OR%20NFLX%20OR%20TSLA%20OR%20META%20OR%20AMZN%20OR%20NVDA%20OR%20GOOG%20OR%20MSFT%20OR%20SHOP%20OR%20AMD%20OR%20UPST%20OR%20AAL%20OR%20TSM%20OR%20Nasdaq%20OR%20Dow)%20lang%3Aen%20until%3A2024-07-01%20since%3A2024-06-01%22%2C%22count%22%3A20%2C%22cursor%22%3A%22${data.data.search_by_raw_query.search_timeline.timeline.instructions[2].entry.content.value}%22%2C%22querySource%22%3A%22typed_query%22%2C%22product%22%3A%22Top%22%7D&features=%7B%22rweb_tipjar_consumption_enabled%22%3Atrue%2C%22responsive_web_graphql_exclude_directive_enabled%22%3Atrue%2C%22verified_phone_label_enabled%22%3Afalse%2C%22creator_subscriptions_tweet_preview_api_enabled%22%3Atrue%2C%22responsive_web_graphql_timeline_navigation_enabled%22%3Atrue%2C%22responsive_web_graphql_skip_user_profile_image_extensions_enabled%22%3Afalse%2C%22communities_web_enable_tweet_community_results_fetch%22%3Atrue%2C%22c9s_tweet_anatomy_moderator_badge_enabled%22%3Atrue%2C%22articles_preview_enabled%22%3Atrue%2C%22tweetypie_unmention_optimization_enabled%22%3Atrue%2C%22responsive_web_edit_tweet_api_enabled%22%3Atrue%2C%22graphql_is_translatable_rweb_tweet_is_translatable_enabled%22%3Atrue%2C%22view_counts_everywhere_api_enabled%22%3Atrue%2C%22longform_notetweets_consumption_enabled%22%3Atrue%2C%22responsive_web_twitter_article_tweet_consumption_enabled%22%3Atrue%2C%22tweet_awards_web_tipping_enabled%22%3Afalse%2C%22creator_subscriptions_quote_tweet_preview_enabled%22%3Afalse%2C%22freedom_of_speech_not_reach_fetch_enabled%22%3Atrue%2C%22standardized_nudges_misinfo%22%3Atrue%2C%22tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled%22%3Atrue%2C%22rweb_video_timestamps_enabled%22%3Atrue%2C%22longform_notetweets_rich_text_read_enabled%22%3Atrue%2C%22longform_notetweets_inline_media_enabled%22%3Atrue%2C%22responsive_web_enhance_cards_enabled%22%3Afalse%7D`;
                }

                // Save tweets to CSV after each request
                await saveToCsv(tweets, i === 0);
            } catch (error) {
                console.error(`Error processing request ${i + 1}:`, error.message);
                console.log('Waiting for 5 minutes before continuing...');
                await delay(300000); // Wait for 5 minutes
            }
        }
    };


    const saveToCsv = async (tweets, isFirstRequest) => {
        const csvContent = stringify(
            tweets.map(tweet => [tweet.date, tweet.text]),
            {
                header: isFirstRequest,
                columns: ['Date', 'Tweet'],
                quoted: true,
                quotedString: true,
                quotedEmpty: true,
            }
        );

        try {
            if (isFirstRequest) {
                // await fs.promises.writeFile('tweets.csv', csvContent, 'utf8');
                await fs.promises.appendFile('tweets.csv', csvContent, 'utf8');
            } else {
                await fs.promises.appendFile('tweets.csv', csvContent, 'utf8');
            }
            console.log('CSV file updated successfully');
        } catch (err) {
            console.error('Error writing to CSV file', err);
        }
    };

    await processTweets(100000); // specify the number of requests
})();