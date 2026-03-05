// GYM PARTNER — Service Worker
// Cache le shell de l'app pour une installation rapide.
// Les requêtes POST (upload vidéo) ne sont jamais interceptées.

const CACHE_NAME = 'gympartner-v1';
const SHELL = ['/', '/index.html'];

// Installation : mise en cache du shell
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL))
  );
  self.skipWaiting();
});

// Activation : nettoyage des anciens caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch : network-first, fallback cache (GET uniquement)
self.addEventListener('fetch', event => {
  // Ne jamais intercepter les POST (upload vidéo, analyse)
  if (event.request.method !== 'GET') return;

  // Ne pas intercepter les appels API
  const url = new URL(event.request.url);
  if (url.pathname.startsWith('/analyze') || url.pathname === '/history') return;

  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Mettre à jour le cache avec la réponse fraîche
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
