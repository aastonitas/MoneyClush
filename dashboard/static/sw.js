// MoneyClush push receiver. Deliberately minimal: it does not intercept
// `fetch` or cache anything, only receives push events and reacts to
// clicks on the notifications they produce. Adding offline behaviour
// later is a separate decision from adding push.

self.addEventListener("push", (event) => {
  let data = { title: "MoneyClush", body: "" };
  try {
    if (event.data) data = event.data.json();
  } catch (e) {
    if (event.data) data.body = event.data.text();
  }

  const options = {
    body: data.body || "",
    tag: data.tag || undefined,
    // Replacing an existing notification with the same tag (e.g. the same
    // BTC window's edge) instead of stacking duplicates.
    renotify: !!data.tag,
    data: { url: data.url || "/" },
  };

  event.waitUntil(
    self.registration.showNotification(data.title || "MoneyClush", options)
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || "/";
  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then((wins) => {
      for (const w of wins) {
        if ("focus" in w) return w.focus();
      }
      if (clients.openWindow) return clients.openWindow(url);
    })
  );
});
