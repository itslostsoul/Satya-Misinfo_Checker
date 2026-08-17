self.addEventListener('fetch', (event) => {
    // Intercept the Android Share POST request
    if (event.request.method === 'POST' && event.request.url.includes('/_share_target')) {
        event.respondWith((async () => {
            try {
                // Extract the image from the incoming payload
                const formData = await event.request.formData();
                const imageBlob = formData.get('image'); 

                // Save the image into the browser's hidden cache
                const cache = await caches.open('satya-share-cache');
                await cache.put('/shared-image', new Response(imageBlob));

                // Instantly redirect the browser to your frontend UI
                return Response.redirect('/?shared=true', 303);
            } catch (error) {
                console.error('Share Intercept Error:', error);
                return Response.redirect('/?error=share_failed', 303);
            }
        })());
    }
});